import numpy as np
import pandas as pd
from datetime import timedelta
import torch
from torch.utils.data import DataLoader
from tsai.all import *
from joblib import  load
from datetime import datetime
from prepare_df import calculate_movement_vars
from modelling_tst import set_every_seed

def align_df_prep(match_df):
    """ Aligns the data by inclduing the ball values in each player row and calculates difference variables.
    
    Parameters:
    match_df -- the DataFrame containing the match data

    Returns:
    aligned_data          -- the match DataFrame after the alignment and difference variable addition 
    timestamp_player_dict -- the dictionary containing the players for each timestamp
    """
    
    ball_df = match_df[match_df["full name"].str.contains("ball")]
    players_df = match_df[~match_df["full name"].str.contains("ball")]

    # Create dict with player names playing as dict value for corresponding timestamp as dict key
    timestamp_player_dict = players_df.groupby("formatted local time")["full name"].apply(list).to_dict()

    players_df = players_df.copy()
    ball_df = ball_df.copy()

    players_df["name"] = players_df["full name"]
    ball_df["name"] = "Ball"

    merge_vars = ["formatted local time", "name", "x in m", "y in m", "speed in m/s", "acceleration in m/s2", "direction of movement in deg", "sin angle", "cos angle"]

    aligned_data = players_df[["formatted local time", "name", "x in m", "y in m", "speed in m/s", "acceleration in m/s2",
                               "direction of movement in deg", "sin angle", "cos angle"]].merge(ball_df[merge_vars], 
                                on="formatted local time", how="left", suffixes=("_player", "_ball"))
    
    aligned_data["difference distance"] = np.sqrt(
        (aligned_data["x in m_player"] - aligned_data["x in m_ball"]) ** 2 +
        (aligned_data["y in m_player"] - aligned_data["y in m_ball"]) ** 2
    )

    aligned_data["difference speed"] = aligned_data["speed in m/s_player"] - aligned_data["speed in m/s_ball"]

    aligned_data["difference acceleration"] = aligned_data["acceleration in m/s2_player"] - aligned_data["acceleration in m/s2_ball"]

    aligned_data["difference direction"] = round((aligned_data["direction of movement in deg_player"] - aligned_data["direction of movement in deg_ball"] + 180) % 360 - 180, 3)

    aligned_data["difference angle rad"] = np.deg2rad(aligned_data["difference direction"])

    # Transform angles into sine and cosine components
    aligned_data["difference sin angle"] = np.sin(aligned_data["difference angle rad"])
    aligned_data["difference cos angle"] = np.cos(aligned_data["difference angle rad"])

    aligned_data.drop(columns=["difference angle rad"], inplace=True)

    return aligned_data, timestamp_player_dict

def pivot_df_prep(aligned_df):
    """ Uses the pivot function to have all rows per timestamp condensed into one row.
    
    Parameters:
    aligned_df -- the aligned DataFrame

    Returns:
    pivot_df -- the match DataFrame after the pivot transformation
    """

    diff_cols = ["formatted local time", "difference distance", "difference speed",
                 "difference acceleration", "difference sin angle", "difference cos angle"]

    # Separate modified player data
    player_data_columns = [col for col in aligned_df.columns if col.endswith("_player") or col in diff_cols]
    player_data = aligned_df[player_data_columns].copy()
    # Rename columns back to original
    player_data.columns = player_data.columns.str.replace("_player", "")

    ball_data_columns = [col for col in aligned_df.columns if col.endswith("_ball") or col in diff_cols]
    ball_data = aligned_df[ball_data_columns].drop_duplicates(subset=["formatted local time"])
    ball_data.columns = ball_data.columns.str.replace("_ball", "")

    reconstructed_df = pd.concat([player_data, ball_data]).sort_values(by="formatted local time")
    reconstructed_df.sort_values(by="formatted local time", inplace=True)
    reconstructed_df.reset_index(drop=True, inplace=True)

    pivot_df = reconstructed_df.pivot_table(index="formatted local time", columns="name", values=["x in m", "y in m", "speed in m/s", "acceleration in m/s2", "sin angle", "cos angle",
                                                                                                  "difference distance", "difference speed", "difference acceleration",
                                                                                                  "difference sin angle", "difference cos angle"])

    # Flatten multi-index columns
    pivot_df.columns = ["_".join(col).strip() for col in pivot_df.columns.values]

    # Extract unique player names and sort them
    players = sorted(set([col[col.find("_") + 1:] for col in pivot_df.columns if not "Ball" in col]))

    # Reorder columns to group corresponding variables for each player
    ordered_columns = ["x in m_Ball", "y in m_Ball", "speed in m/s_Ball", "acceleration in m/s2_Ball", "sin angle_Ball", "cos angle_Ball"]
    for player in players:
        ordered_columns.extend([f"x in m_{player}", f"y in m_{player}", f"speed in m/s_{player}", f"acceleration in m/s2_{player}",
                                f"sin angle_{player}", f"cos angle_{player}", f"difference distance_{player}", f"difference speed_{player}",
                                f"difference acceleration_{player}", f"difference sin angle_{player}", f"difference cos angle_{player}"])

    pivot_df = pivot_df[ordered_columns]
    pivot_df.reset_index(inplace=True)
    
    return pivot_df

def sample_retrieval(player, pivot_df, window_start, window_end, input_time_steps):
    """ Retrieves the sample values from the specified time window.
    
    Parameters:
    player        -- the name of the current player within the timestamp
    pivot_df      -- the pivot transformed DataFrame
    window_start  -- the starting point of the window
    window_end    -- the ending point of the window

    Returns:
    df_timestamp  -- the DataFrame for the current timestamp window
    label         -- the label indicating if the player is in possession
    usable_sample -- the string stating wether there is enough data to make the sample usable
    """

    usable_sample = "true"
    df_timestamp = pivot_df[(pivot_df["formatted local time"] >= window_start) & 
                            (pivot_df["formatted local time"] <= window_end)]

    df_len = len(df_timestamp)

    df_timestamp = df_timestamp[[f"difference distance_{player}", "x in m_Ball", "y in m_Ball", f"x in m_{player}", f"y in m_{player}",
                                 "speed in m/s_Ball", f"speed in m/s_{player}", f"difference speed_{player}",
                                 "acceleration in m/s2_Ball", f"acceleration in m/s2_{player}", f"difference acceleration_{player}",
                                 "sin angle_Ball", "cos angle_Ball", f"sin angle_{player}", f"cos angle_{player}",
                                 f"difference sin angle_{player}", f"difference cos angle_{player}"]].to_numpy()
    if (df_len < input_time_steps) or (np.isnan(df_timestamp).any() == True):
        usable_sample = "false"

    return df_timestamp, usable_sample

def create_samples(timestamp_game, data_amount, pivot_df, timestamp_player_dict, window_length_datetime, input_time_steps):
    """ Creates the complete samples for the train/val/test set.
    
    Parameters:
    timestamp_game          -- the DataFrame the pivot function should be performed on
    data_amount             -- the amount of samples to be considered
    pivot_df                -- the pivot transformed DataFrame
    timestamp_player_dict   -- the dictionary containing the players for each timestamp

    Returns:
    tsd           -- the TimeseriesDataset containing the X and y tensors
    ts_list_total -- the list containing the used timestamps
    """

    timestamps = pd.to_datetime(timestamp_game) # Get timestamps from timestamp_game combination for window calculations
    window_starts = timestamps - window_length_datetime # Pre-calculate start and end times for each window

    X_list_total = []
    ts_list_total = [] # List for saving used timestamps for being being able to measure prediction performance

    for window_start, window_end in tqdm(zip(window_starts[:data_amount], timestamps[:data_amount])):
        X_list_temporary = []
        ts_list_temporary = [] # List for saving used timestamps for being able to measure prediction performance
        usable_sample = "true"
        if pd.Timestamp(window_end) in timestamp_player_dict:
            for player in timestamp_player_dict[pd.Timestamp(window_end)]:
                sample, usable_sample = sample_retrieval(player, pivot_df, window_start, window_end, input_time_steps=input_time_steps)
                if usable_sample == "false":
                    break # Skip timestamp if not enough data is available

                ts_list_temporary.append(window_end)
                X_list_temporary.append(sample)
            if usable_sample == "true":
                X_list_total += X_list_temporary
                ts_list_total += ts_list_temporary
        else:
            pass

    # Convert lists to numpy arrays
    X = np.array(X_list_total)

    X = X.transpose(0, 2, 1) # Transpose to have correct ordering for TST model 

    X_tensor = torch.tensor(X, dtype=torch.float32)

    print("Sizes", X_tensor.size())

    tsd = TSDatasets(X_tensor)

    return tsd, ts_list_total

set_every_seed()

window_length_dt = timedelta(milliseconds=1000)
input_time_steps = 21

match_full = pd.read_parquet("match_w_inference_w_metadata_df.parquet")
match_full["timestamp"] = pd.to_datetime(match_full["timestamp"], unit="ms")
match_full["name"] = match_full["name"].str.lower().str.replace(" ", "_")
match_full_original = match_full.copy()

match_full.rename(columns={"x": "x in m", "y": "y in m", "timestamp": "formatted local time", "name": "full name"}, inplace=True)

# Remove every player sitting on bench and every ball not in use
match_full.drop(match_full[match_full["y in m"] < -10].index, inplace=True)
match_full = calculate_movement_vars(match_full) # Add movement variables
match_full = match_full.dropna(subset=["x in m", "y in m", "speed in m/s", "acceleration in m/s2", "direction of movement in deg"], axis=0)

aligned_df, ts_player_dict = align_df_prep(match_full)

columns_to_scale = [# Player
                    "x in m_player", "y in m_player",
                    "speed in m/s_player", "acceleration in m/s2_player",
                    "sin angle_player", "cos angle_player",
                    # Ball
                    "x in m_ball", "y in m_ball",
                    "speed in m/s_ball", "acceleration in m/s2_ball",
                    "sin angle_ball", "cos angle_ball",
                    # Differences
                    "difference distance", "difference speed", "difference acceleration",
                    "difference sin angle", "difference cos angle"]
columns_not_to_scale = [col for col in aligned_df.columns if col not in columns_to_scale]
scaler = load(f"handball_sample\scaler_1000ms.joblib")
df_scaled = pd.DataFrame(scaler.transform(aligned_df[columns_to_scale]), columns=columns_to_scale, index=aligned_df.index)
match_df_scaled = pd.concat([df_scaled, aligned_df[columns_not_to_scale]], axis=1)

pivot_df = pivot_df_prep(match_df_scaled)

timestamps_test = match_full["formatted local time"].unique()
print(len(timestamps_test))

test_ds, test_timestamps = create_samples(timestamp_game=timestamps_test, data_amount=len(timestamps_test), pivot_df=pivot_df, timestamp_player_dict = ts_player_dict,
                                          window_length_datetime=window_length_dt, input_time_steps=input_time_steps)

# Load best params
params_tst = pd.read_csv(f"handball_sample/best_params_tst_1000ms_3.csv", index_col=0)
batch_size = params_tst["batch_size"][0]

learner_test = load_learner(f"handball_sample/tst_model_1000ms.pth", cpu=False)

# Create DataLoader from test dataset
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

start_time_get_preds = datetime.now()
preds, y_true = learner_test.get_preds(dl=test_dl)
end_time_get_preds = datetime.now()
print("preds", end_time_get_preds - start_time_get_preds)

# Final accuracy over all timestamps
unique = list(dict.fromkeys(test_timestamps))
predicted_df = match_full[match_full["formatted local time"].isin(unique)]
predicted_df = predicted_df.drop(predicted_df[predicted_df["full name"].str.contains("ball")].index, axis=0)
predicted_df["possession_pred_prob"] = preds

# Identify highest probability for possession within each timestamp
predicted_df["max_flag"] = predicted_df.groupby("formatted local time")["possession_pred_prob"].transform(lambda x: (x == x.max()).astype(int))

# Add possession values to original df
match_full_poss_update = match_full_original.merge(predicted_df[["formatted local time", "full name", "max_flag"]], how="left", left_on=["timestamp", "name"], right_on=["formatted local time", "full name"])
has_poss_df = match_full_poss_update[match_full_poss_update["max_flag"] == 1]
match_full_poss_update = match_full_poss_update.merge(has_poss_df[['timestamp', 'name']], how="left", on="timestamp", suffixes=("", "_in_possession"))
match_full_poss_update["name_in_possession"] = match_full_poss_update["name_in_possession"].ffill()
match_full_poss_update["ball_possession_updated"] = match_full_poss_update["name"] == match_full_poss_update["name_in_possession"]
print(match_full_poss_update["ball_possession_updated"].value_counts())

match_full_poss_update.to_parquet(f"match_w_inference_w_metadata_df_updated_poss.parquet")