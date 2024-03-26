import numpy as np
import pandas as pd
from datetime import timedelta
from tqdm import tqdm
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader
from tsai.all import *
import optuna
from optuna.integration import FastAIPruningCallback

window_length_ms, tuning = int(sys.argv[1]), str(sys.argv[2])

window_length_datetime = timedelta(milliseconds=window_length_ms)
input_time_steps = (window_length_ms / 50) + 1 # Amount of input timesteps for model, depending on window length

def set_every_seed(seed=42):
    """ Sets every possible seed.
    
    Parameters:
    seed -- the seed to be set

    Returns:
    None
    """

    set_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_every_seed()

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
                               "direction of movement in deg", "sin angle", "cos angle", "tag text", "possession"]].merge(ball_df[merge_vars], 
                                on="formatted local time", how="left", suffixes=("_player", "_ball"))
    
    aligned_data["difference distance"] = np.sqrt(
        (aligned_data["x in m_player"] - aligned_data["x in m_ball"]) ** 2 +
        (aligned_data["y in m_player"] - aligned_data["y in m_ball"]) ** 2
    )

    aligned_data["difference speed"] = aligned_data["speed in m/s_player"] - aligned_data["speed in m/s_ball"]

    aligned_data["difference acceleration"] = aligned_data["acceleration in m/s2_player"] - aligned_data["acceleration in m/s2_ball"]

    aligned_data["difference direction"] = round((aligned_data["direction of movement in deg_player"] - aligned_data["direction of movement in deg_ball"] + 180) % 360 - 180, 3)

    aligned_data['difference angle rad'] = np.deg2rad(aligned_data['difference direction'])

    # Transform angles into sine and cosine components
    aligned_data['difference sin angle'] = np.sin(aligned_data['difference angle rad'])
    aligned_data['difference cos angle'] = np.cos(aligned_data['difference angle rad'])

    aligned_data.drop(columns=["difference angle rad"], inplace=True)

    return aligned_data, timestamp_player_dict

def pivot_df_prep(aligned_df):
    """ Uses the pivot function to have all rows per timestamp condensed into one row.
    
    Parameters:
    aligned_df -- the aligned DataFrame

    Returns:
    pivot_df -- the match DataFrame after the pivot transformation
    """

    diff_cols = ["formatted local time", "tag text", "possession", "difference distance", "difference speed",
                 "difference acceleration", "difference sin angle", "difference cos angle"]

    # Separate modified player data
    player_data_columns = [col for col in aligned_df.columns if col.endswith("_player") or col in diff_cols]
    player_data = aligned_df[player_data_columns].copy()
    # Rename columns back to original
    player_data.columns = player_data.columns.str.replace("_player", "")

    ball_data_columns = [col for col in aligned_df.columns if col.endswith("_ball") or col in diff_cols]
    ball_data = aligned_df[ball_data_columns].drop_duplicates(subset=['formatted local time'])
    ball_data.columns = ball_data.columns.str.replace("_ball", "")

    reconstructed_df = pd.concat([player_data, ball_data]).sort_values(by="formatted local time")
    reconstructed_df.sort_values(by="formatted local time", inplace=True)
    reconstructed_df.reset_index(drop=True, inplace=True)

    pivot_df = reconstructed_df.pivot_table(index="formatted local time", columns="name", values=["x in m", "y in m", "speed in m/s", "acceleration in m/s2", "sin angle", "cos angle",
                                                                                                  "difference distance", "difference speed", "difference acceleration",
                                                                                                  "difference sin angle", "difference cos angle", "possession"])

    # Flatten multi-index columns
    pivot_df.columns = ["_".join(col).strip() for col in pivot_df.columns.values]

    # Extract unique player names and sort them
    players = sorted(set([col[col.find("_") + 1:] for col in pivot_df.columns if not "Ball" in col]))

    # Reorder columns to group corresponding variablers for each player
    ordered_columns = ["x in m_Ball", "y in m_Ball", "speed in m/s_Ball", "acceleration in m/s2_Ball", "sin angle_Ball", "cos angle_Ball"]
    for player in players:
        ordered_columns.extend([f"x in m_{player}", f"y in m_{player}", f"possession_{player}", f"speed in m/s_{player}", f"acceleration in m/s2_{player}",
                                f"sin angle_{player}", f"cos angle_{player}", f"difference distance_{player}", f"difference speed_{player}",
                                f"difference acceleration_{player}", f"difference sin angle_{player}", f"difference cos angle_{player}"])

    pivot_df = pivot_df[ordered_columns]
    pivot_df.reset_index(inplace=True)
    
    return pivot_df

def sample_retrieval(player, pivot_df, window_start, window_end):
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
    label = pivot_df[pivot_df["formatted local time"] == window_end][f"possession_{player}"]
    if (df_len < input_time_steps) or (np.isnan(df_timestamp).any() == True):
        usable_sample = "false"

    return df_timestamp, label, usable_sample

def create_samples(timestamp_game, data_amount, pivot_df, timestamp_player_dict):
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
    y_list_total = []
    ts_list_total = [] # List for saving used timestamps for being being able to measure prediction performance 

    for window_start, window_end in tqdm(zip(window_starts[:data_amount], timestamps[:data_amount])):
        X_list_temporary = []
        y_list_temporary = []
        ts_list_temporary = [] # List for saving used timestamps for being able to measure prediction performance
        usable_sample = "true"
        for player in timestamp_player_dict[pd.Timestamp(window_end)]:
            sample, label, usable_sample = sample_retrieval(player, pivot_df, window_start, window_end)
            if usable_sample == "false":
                break # Skip timestamp if not enough data is available

            ts_list_temporary.append(window_end)
            X_list_temporary.append(sample)
            y_list_temporary.append(label)
        if usable_sample == "true":
            X_list_total += X_list_temporary
            y_list_total += y_list_temporary
            ts_list_total += ts_list_temporary

    # Convert lists to numpy arrays
    X = np.array(X_list_total)
    y = np.array(y_list_total)

    X = X.transpose(0, 2, 1) # Transpose to have correct ordering for TST model 

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    tsd = TSDatasets(X_tensor, y_tensor)

    return tsd, ts_list_total

def objective(trial:optuna.Trial):
    """ Creates the search space for hyperparameter tuning and executes the tuning.
    
    Parameters:
    trial -- the current Optuna trial object

    Returns:
    best_valid_loss -- the best validation loss value
    """

    number_of_epochs = 10
    early_stopping = True # Boolean stating wether early stopping should be used during tuning

    # Search space
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    dropout = trial.suggest_float("dropout", 0.0, 0.3, step=0.1)
    fc_dropout = trial.suggest_float("fc_dropout", 0.0, 0.8, step=0.1)
    n_layers = trial.suggest_categorical("n_layers", [2, 3, 4, 5, 6, 7, 8])
    d_model = trial.suggest_categorical("d_model", [128, 256, 512, 1024])
    n_heads = trial.suggest_categorical("n_heads", [2, 8, 10, 12, 14, 16])
    d_k = trial.suggest_categorical("d_k", [8, 16, 32, 64, 128, 256, 512])
    d_v = trial.suggest_categorical("d_v", [8, 16, 32, 64, 128, 256, 512])
    d_ff = trial.suggest_categorical("d_ff", [256, 512, 1024, 2048, 4096])

    # Give positive class more weight
    num_class0 = 13
    num_class1 = 1
    total = num_class0 + num_class1
    weight_class0 = total / (2.0 * num_class0)
    weight_class1 = total / (2.0 * num_class1)
    class_weights = torch.tensor([weight_class0, weight_class1])

    dls = TSDataLoaders.from_dsets(train_ds, val_ds, bs=batch_size, tfms=[None, TSClassification()], num_workers=0)
    model = TST(c_in=dls.vars, c_out=dls.c, seq_len=dls.len, n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=dropout, fc_dropout=fc_dropout)
    learn = Learner(dls, model, loss_func=nn.BCEWithLogitsLoss(pos_weight=class_weights[1]),
                    metrics=[F1ScoreMulti(), RocAucMulti(), PrecisionMulti(), RecallMulti()], cbs=FastAIPruningCallback(trial, monitor="valid_loss"))
    
    if early_stopping == False:
        # with ContextManagers([learn.no_logging(), learn.no_bar()]): # Prevents printing anything during training
        learn.fit_one_cycle(number_of_epochs, lr_max=learning_rate)
    else:
        best_f1_score = 0
        epochs_since_improvement = 0
        patience = 5
        delta = 0.05

        for _ in range(number_of_epochs):
            #with ContextManagers([learn.no_logging(), learn.no_bar()]):
            learn.fit_one_cycle(1, lr_max = learning_rate)
            current_f1_score = learn.recorder.values[-1][2]

            if best_f1_score - current_f1_score > delta:
                best_f1_score = current_f1_score
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= patience:
                raise optuna.exceptions.TrialPruned()  # Prune trial if patience is exceeded and not enough improvement has been made

    return best_f1_score

# Load data
match_train_val_df = pd.read_csv(r"handball_sample\match_training_model.csv", sep=";", index_col=0)
match_test_df = pd.read_csv(r"handball_sample\match_test_model.csv", sep=";", index_col=0)

match_train_val_df["formatted local time"] = pd.to_datetime(match_train_val_df["formatted local time"])
match_test_df["formatted local time"] = pd.to_datetime(match_test_df["formatted local time"])

match_train_df = match_train_val_df[match_train_val_df["game"].isin(["FLEvsKIE", "ERLvsFLE", "FLEvsEIS"])]
match_val_df = match_train_val_df[match_train_val_df["game"].isin(["GUMvsFLE"])]

match_test_df.reset_index(inplace=True)

print("Aligning DataFrames")
aligned_df_train, ts_player_dict_train = align_df_prep(match_train_df)
aligned_df_val, ts_player_dict_val = align_df_prep(match_val_df)
aligned_df_test, ts_player_dict_test = align_df_prep(match_test_df)

# Columns to be scaled
columns_to_scale = ['x in m_player', 'y in m_player',
       'speed in m/s_player', 'acceleration in m/s2_player',
       'sin angle_player', "cos angle_player",
       'x in m_ball', 'y in m_ball', 'speed in m/s_ball',
       'acceleration in m/s2_ball',
       'sin angle_ball', "cos angle_ball",
       'difference distance', 'difference speed', 'difference acceleration',
       'difference sin angle', "difference cos angle"]

# Columns not to scale
columns_not_to_scale = [col for col in aligned_df_train.columns if col not in columns_to_scale]
# Fit the scaler on the training data (only on columns to scale)
scaler = StandardScaler().fit(aligned_df_train[columns_to_scale])
# Transform the data (only scale the columns that need scaling)
df_train_scaled = pd.DataFrame(scaler.transform(aligned_df_train[columns_to_scale]), columns=columns_to_scale, index=aligned_df_train.index)
df_val_scaled = pd.DataFrame(scaler.transform(aligned_df_val[columns_to_scale]), columns=columns_to_scale, index=aligned_df_val.index)
df_test_scaled = pd.DataFrame(scaler.transform(aligned_df_test[columns_to_scale]), columns=columns_to_scale, index=aligned_df_test.index)
# Concatenate the scaled columns back with the columns that were not scaled
match_train_df_scaled = pd.concat([df_train_scaled, aligned_df_train[columns_not_to_scale]], axis=1)
match_val_df_scaled = pd.concat([df_val_scaled, aligned_df_val[columns_not_to_scale]], axis=1)
match_test_df_scaled = pd.concat([df_test_scaled, aligned_df_test[columns_not_to_scale]], axis=1)

# Change DataFrame structure
print("Pivoting DataFrames")
pivot_df_train = pivot_df_prep(match_train_df_scaled)
pivot_df_val = pivot_df_prep(match_val_df_scaled)
pivot_df_test = pivot_df_prep(match_test_df_scaled)

# Only use visible timestamps
visible_df_train = match_train_df[(match_train_df["tag text"] != "not_visible")]
visible_df_val = match_val_df[(match_val_df["tag text"] != "not_visible")]
visible_df_test = match_test_df[(match_test_df["tag text"] != "not_visible")]

# Get unique timestamps
timestamps_train = visible_df_train["formatted local time"].unique()
timestamps_val = visible_df_val["formatted local time"].unique()
timestamps_test = visible_df_test["formatted local time"].unique()

random.seed(42)
random.shuffle(timestamps_train)
random.shuffle(timestamps_val)

# Prepare train, val and test data
print("Creating samples")
train_ds, _ = create_samples(timestamp_game=timestamps_train, data_amount=20000, pivot_df=pivot_df_train, timestamp_player_dict = ts_player_dict_train)
val_ds, _ = create_samples(timestamp_game=timestamps_val, data_amount=7000, pivot_df=pivot_df_val, timestamp_player_dict = ts_player_dict_val)
test_ds, test_timestamps = create_samples(timestamp_game=timestamps_test, data_amount=2000, pivot_df=pivot_df_test, timestamp_player_dict = ts_player_dict_test)

if tuning == "True":
    print("Starting Optuna study")
    study = optuna.create_study(direction="maximize") # Minimize validation loss
    study.optimize(objective, n_trials=100)
    
    # Print study statistics
    print("Study statistics:")
    print("  Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    trial = study.best_trial

    # Create DataFrame with best params and save it
    params_df = pd.DataFrame(trial.params, index=[0])
    params_df.to_csv(r"handball_sample\best_params_tst.csv")
else:
    # Load best params
    params_tst = pd.read_csv(r"handball_sample\best_params_tst.csv", index_col=0)

    # Retrieve best params data
    learning_rate = float(params_tst["learning_rate"][0]) # Learning rate must be converted to float from numpy.float64 due to error
    batch_size = params_tst["batch_size"][0]
    dropout = params_tst["dropout"][0]
    fc_dropout = params_tst["fc_dropout"][0]
    n_layers = params_tst["n_layers"][0]
    d_model = params_tst["d_model"][0]
    n_heads = params_tst["n_heads"][0]
    d_k = params_tst["d_k"][0]
    d_v = params_tst["d_v"][0]
    d_ff = params_tst["d_ff"][0]

    # Give positive class more weight
    num_class0 = 12.5
    num_class1 = 1
    total = num_class0 + num_class1
    weight_class0 = total / (2.0 * num_class0)
    weight_class1 = total / (2.0 * num_class1)
    class_weights = torch.tensor([weight_class0, weight_class1])

    # Final model with best params
    print("Training final model")
    dls = TSDataLoaders.from_dsets(train_ds, val_ds, bs=batch_size, tfms=[None, TSClassification()], num_workers=0)
    dls = dls.to(device)
    model = TST(c_in=dls.vars, c_out=dls.c, seq_len=dls.len, n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=dropout, fc_dropout=fc_dropout)
    learn = Learner(dls, model, loss_func=nn.BCEWithLogitsLoss(pos_weight=class_weights[1]), metrics=[F1ScoreMulti(), RocAucMulti(), PrecisionMulti(), RecallMulti()], cbs=None)
    
    number_of_epochs = 10
    best_f1_score = 0
    epochs_since_improvement = 0
    patience = 5
    delta = 0.03

    for epoch in range(number_of_epochs):
        print("Epoch: ", epoch)
        learn.fit_one_cycle(1, lr_max=learning_rate)

        current_f1_score = learn.recorder.values[-1][2]

        if current_f1_score > best_f1_score + delta:
            best_f1_score = current_f1_score
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            print("Early stopping")
            break

    learn.export(f"handball_sample/tst_model_{window_length_ms}.pth") # Save final model
    
    learner_test = load_learner(f"tst_model_{window_length_ms}.pth", cpu=False)

    # Create DataLoader from test dataset
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Evaluate performance on test data
    loss, f1, rocAuc, prec, rec = learner_test.validate(dl=test_dl)
    print(f"Test Loss: {loss}, Test F1: {f1}, Test ROC AUC: {rocAuc}, Test Precision: {prec}, Test Recall: {rec}")

    preds, y_true = learner_test.get_preds(dl=test_dl)

    # Final accuracy over all timestamps
    unique = list(dict.fromkeys(test_timestamps))
    predicted_df = match_test_df[match_test_df["formatted local time"].isin(unique)]
    predicted_df = predicted_df.drop(predicted_df[predicted_df["full name"].str.contains("ball")].index, axis=0)
    predicted_df["possession_pred_prob"] = preds
    # Identify highest probability for possession within each timestamp
    predicted_df["max_flag"] = predicted_df.groupby("formatted local time")["possession_pred_prob"].transform(lambda x: (x == x.max()).astype(int))
    predicted_df["correct"] = (predicted_df["max_flag"] == predicted_df["possession"]).astype(int)
    total_timestamps = predicted_df["formatted local time"].nunique()
    num_timestamps_all_correct = predicted_df.groupby("formatted local time")["correct"].all().sum()
    print(f"Number of correct timestamps: {num_timestamps_all_correct}, number of total timestamps: {total_timestamps}, Accuracy: {num_timestamps_all_correct / total_timestamps}")