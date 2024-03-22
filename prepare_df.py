import pandas as pd
import numpy as np
from datetime import timedelta
import time_check_df

def main(preparation_type, game, stats):
    def calculate_movement_vars(df):
        """ Calculates the speed, acceleration and direction of movement .
        
        Parameters:
        df -- the DataFrame for which the variables should be calculated

        Returns:
        df -- the DataFrame containing the added movement variables
        """

        # Group by player to have correct calculation
        df["delta_x"] = df.groupby("full name")["x in m"].diff()
        df["delta_y"] = df.groupby("full name")["y in m"].diff()
        
        df["delta_time"] = df.groupby("full name")["formatted local time"].diff().dt.total_seconds()
        
        df["distance"] = np.sqrt(df["delta_x"]**2 + df["delta_y"]**2)
        df["speed in m/s"] = round(df["distance"] / df["delta_time"], 3)
        
        df["delta_speed"] = df.groupby("full name")["speed in m/s"].diff()
        df['acceleration in m/s2'] = df['delta_speed'] / df['delta_time']

        df["direction of movement in deg"] = round(np.degrees(np.arctan2(df["delta_y"], df["delta_x"])), 3)

        df['angle rad'] = np.deg2rad(df['direction of movement in deg'])
        df['sin angle'] = np.sin(df['angle rad'])
        df['cos angle'] = np.cos(df['angle rad'])
        
        # Set value to nan if time difference is not exactly 0.05, e.g. due to playing stoppage 
        df.loc[df["delta_time"] != 0.05, ["speed in m/s", "acceleration in m/s2", "sin angle", "cos angle"]] = np.nan
        
        df.drop(columns=["delta_x", "delta_y", "delta_speed", "delta_time", "distance"], inplace=True)

        return df

    offsets = {"FLEvsKIE": 500,  "ERLvsFLE": 300, "FLEvsEIS": 400, "FLEvsRNL": None, "GUMvsFLE": -1300, "FLEvsMEL": 350} # Offset values for tags, depend on comparison in time_check_df

    if stats == "True":
        file_name_add = "stats"
    else:
        file_name_add = "model"

    if preparation_type == "single":
        # Load data 
        if game == "FLEvsKIE":
            match_h1 = pd.read_csv(r"handball_sample\MD05_Flensburg_Kiel\SG_Flensburg-Handewitt_vs._THW_phase_1.HZ_positions.csv", sep=";")
            match_h2 = pd.read_csv(r"handball_sample\MD05_Flensburg_Kiel\SG_Flensburg-Handewitt_vs._THW_phase_2.HZ_positions.csv", sep=";")
            tags = pd.read_csv(r"handball_sample\MD05_Flensburg_Kiel\tags_flensburg_kiel_md5_s2324.csv", sep=";")
        elif game == "ERLvsFLE":
            match_h1 = pd.read_csv(r"handball_sample\MD10_Erlangen_Flensburg\HC_Erlangen_vs._SG_Flensburg-H_phase_1._HZ_positions.csv", sep=";")
            match_h2 = pd.read_csv(r"handball_sample\MD10_Erlangen_Flensburg\HC_Erlangen_vs._SG_Flensburg-H_phase_2._HZ_positions.csv", sep=";")
            tags = pd.read_csv(r"handball_sample\MD10_Erlangen_Flensburg\tags_erlangen_flensburg_md10_s2324.csv", sep=";")
        elif game == "FLEvsEIS":
            match_h1 = pd.read_csv(r"handball_sample\MD11_Flensburg_Eisenach\SG_Flensburg-Handewitt_vs._ThS_phase_1._Halbzeit__positions.csv", sep=";")
            match_h2 = pd.read_csv(r"handball_sample\MD11_Flensburg_Eisenach\SG_Flensburg-Handewitt_vs._ThS_phase_2._Halbzeit__positions.csv", sep=";")
            tags = pd.read_csv(r"handball_sample\MD11_Flensburg_Eisenach\tags_flensburg_eisenach_md11_s2324.csv", sep=";")
        # elif game == "FLEvsRNL":
        #     match_h1 = pd.read_csv(r"handball_sample\MD13_Flensburg_RNL\SG_Flensburg-Handewitt_vs._Rhe_phase_1._Halbzeit__positions.csv", sep=";")
        #     match_h2 = pd.read_csv(r"handball_sample\MD13_Flensburg_RNL\SG_Flensburg-Handewitt_vs._Rhe_phase_2._Halbzeit__positions.csv", sep=";")
        #     tags = pd.read_csv(r"handball_sample\MD13_Flensburg_RNL\tags_flensburg_rnl_md13_s2324.csv", sep=";")
        elif game == "GUMvsFLE":
            match_h1 = pd.read_csv(r"handball_sample\MD14_Gummersbach_Flensburg\VfL_Gummersbach_vs._SG_Flensbu_phase_1._Halbzeit_positions.csv", sep=";")
            match_h2 = pd.read_csv(r"handball_sample\MD14_Gummersbach_Flensburg\VfL_Gummersbach_vs._SG_Flensbu_phase_2._Halbzeit_positions.csv", sep=";")
            tags = pd.read_csv(r"handball_sample\MD14_Gummersbach_Flensburg\tags_gummersbach_flensburg_md14_s2324.csv", sep=";")
        elif game == "FLEvsMEL":
            match_h1 = pd.read_csv(r"handball_sample\MD15_Flensburg_Melsungen\SG_Flensburg-Handewitt_vs._MT__phase_1.HZ_positions.csv", sep=";")
            match_h2 = pd.read_csv(r"handball_sample\MD15_Flensburg_Melsungen\SG_Flensburg-Handewitt_vs._MT__phase_2.HZ_positions.csv", sep=";")
            tags = pd.read_csv(r"handball_sample\MD15_Flensburg_Melsungen\tags_flensburg_melsungen_md15_s2324.csv", sep=";")

        match_full = pd.concat([match_h1, match_h2]).reset_index().drop(["index", "heart rate in bpm", "core temperature in celsius", "player orientation in deg",  "Unnamed: 23"], axis=1)
        match_full["formatted local time"] = pd.to_datetime(match_full["formatted local time"])

        match_start = match_full[match_full[~np.isnan(match_full["ball possession (id of possessed ball)"])].index[0]:]
        match_start.loc[:, "time diff from start"] = match_start.loc[:, "formatted local time"] - match_start["formatted local time"].iloc[0]
        match_start_poss = match_start.dropna(subset=["ball possession (id of possessed ball)"])
        # Drop consecutive rows where same player is still in possession, this is achieved by comparing each row with previous row and keeping only rows where player in possession changes
        match_start_poss = match_start_poss[match_start_poss["sensor id"] != match_start_poss["sensor id"].shift(1)]

        # Get game start time
        if game == "FLEvsEIS":
            table_start_time = pd.to_datetime("2023-10-28 19:02:14.000") # No player has possession in first half of that game according to Kinexon
        # elif game == "FLEvsRNL":
        #     table_start_time = pd.to_datetime("2023-11-18 18:02:10.650") # Possession starts to late
        else:
            table_start_time = match_start_poss["formatted local time"].iloc[0]

        tags.drop(["player"], axis=1, inplace=True)
        tags["# time (in ms) vid"] = tags["# time (in ms)"]
        tags["# time (in ms)"] = tags["# time (in ms)"] - tags["# time (in ms)"][tags["tag text"] == "game_start"].iloc[0]
        
        # Create adjusted timestamp column where video time of tags is converted to actual timestamp 
        tags["adjusted_timestamp"] = tags["# time (in ms)"].apply(lambda x: table_start_time + timedelta(milliseconds=x + offsets[game]))
        tags["rounded_timestamp"] = tags["adjusted_timestamp"].apply(time_check_df.round_to_nearest_50ms) # Round adjusted timestamps

        # Forward fill tags df to have correct merging of match and tags df
        video_start_time = tags["rounded_timestamp"].iloc[0]
        table_end_time = match_full["formatted local time"].iloc[-1]

        # Create timestamp df with frequency of 20Hz (frequency of Kinexon data) from game start to end time
        all_timestamps = pd.date_range(start=video_start_time, end=table_end_time, freq="50L")
        all_timestamps_df = pd.DataFrame(all_timestamps, columns=["timestamp"])

        # Merge tags with all timestamps and forward fill
        tags_full = pd.merge(all_timestamps_df, tags, left_on="timestamp", right_on="rounded_timestamp", how="left")
        tags_full["tag text"].replace(["game_start", "no_possession", "time_check_position"], np.nan, inplace=True)
        cols_from_tags = tags.columns.difference(["# time (in ms)", "adjusted_timestamp"])
        for col in cols_from_tags:
            tags_full[col] = tags_full[col].ffill()
        
        # Remove every player sitting on bench and every ball not in use
        match_full.drop(match_full[match_full["y in m"] < -10].index, inplace=True)
        
        match_full = calculate_movement_vars(match_full) # Add movement variables

        match_full = match_full.dropna(subset = ["x in m", "y in m", "speed in m/s", "acceleration in m/s2", "direction of movement in deg"], axis = 0)

        if stats == "False":
            # Only use timestamps where exactly one ball and 12-14 players are present
            group_counts = match_full.groupby("formatted local time")["group name"].agg(ball_count=lambda x: (x == "Ball").sum(), player_count=lambda x: (x != "Ball").sum()).reset_index()
            filtered_timestamps = group_counts[(group_counts["ball_count"] == 1) & (group_counts["player_count"] >= 12) & (group_counts["player_count"] <= 14)]["formatted local time"]
            filtered_match = match_full[match_full["formatted local time"].isin(filtered_timestamps)]
            filtered_match.loc[:, "formatted local time"] = pd.to_datetime(filtered_match["formatted local time"])

            filtered_merged_df = pd.merge(filtered_match, tags_full, left_on="formatted local time", right_on="timestamp", how="left")
            
            filtered_merged_df.drop(["ts in ms", "sensor id", "mapped id", "league id", "group id",
                                "total distance in m", "timestamp", "# time (in ms)", "adjusted_timestamp"], axis=1, inplace=True)
            prepared_match = filtered_merged_df
        else:
            match_full = pd.merge(match_full, tags_full, left_on="formatted local time", right_on="timestamp", how="left")
            match_full.drop(["ts in ms", "sensor id", "mapped id", "league id", "group id",
                                "total distance in m", "timestamp", "# time (in ms)", "adjusted_timestamp"], axis=1, inplace=True)
            prepared_match = match_full

        # Prepare columns for possession classification
        prepared_match["full name"] = prepared_match["full name"].str.lower().str.replace(" ", "_")
        prepared_match["tag text"] = prepared_match["tag text"].apply(lambda x: x[x.find("_") + 1:])

        # Correct player names
        # Kiel
        prepared_match["tag text"].replace("elias_eliefsen_á_skipagotu", "elias_ellefsen_á_skipagøtu", inplace=True)
        prepared_match["tag text"].replace("kevin_moller", "kevin_møller", inplace=True)
        prepared_match["tag text"].replace("petter_overby", "petter_øverby", inplace=True)
        prepared_match["full name"].replace("mads__mensah_larsen", "mads_mensah_larsen", inplace=True)
        prepared_match["full name"].replace("simon__pytlick", "simon_pytlick", inplace=True)
        # Erlangen
        prepared_match["tag text"].replace("lasse_moller", "lasse_møller", inplace=True)
        prepared_match["full name"].replace("gedeón_guardiola_", "gedeón_guardiola", inplace=True)
        prepared_match["tag text"].replace("mads-peter_lonborg", "mads-peter_lønborg", inplace=True)
        prepared_match["tag text"].replace("nicolai_link", "nikolai_link", inplace=True)
        # Eisenach
        # no change needed
        # RNL

        # Gummersbach
        prepared_match["tag text"].replace("ellidi_vidarsson", "ellidi_snaer_vidarsson", inplace=True)
        prepared_match["tag text"].replace("teitur_orn_einarsson", "teitur_örn_einarsson", inplace=True)
        # Melsungen
        prepared_match["full name"].replace("rogerio__moraes_ferreira", "rogerio_moraes_ferreira", inplace=True)
        prepared_match["tag text"].replace("erik_balenciaga_azcue", "erik_balenciaga", inplace=True)
        prepared_match["tag text"].replace("sindre_andre_aho", "sindre_aho", inplace=True)

        # Check if all players of tag text match full name
        print("Difference between tag_text names and full_name names: ", set(prepared_match["tag text"].unique()) ^ set(prepared_match["full name"].unique()))

        # Create possession column based on tag text, showing 1 for player in possession and otherwise 0
        prepared_match["possession"] = 0
        for index, row in prepared_match.iterrows():
            if row["tag text"] == "not_visible":
                prepared_match.at[index, "possession"] = -1 # Set value to -1 if possession can not be reliably stated in timestamp
            if row["full name"] == row["tag text"]:
                prepared_match.at[index, "possession"] = 1

        if stats == "False":
            # Drop timestamps with no possession in it
            # Group by timestamp and check if all possession values are 0
            timestamps_with_all_zeros = prepared_match.groupby("formatted local time")["possession"].apply(lambda x: x.eq(0).all())
            # Filter timestamps where condition is True
            result_timestamps = timestamps_with_all_zeros[timestamps_with_all_zeros].index
            # Filter out these timestamps from original DataFrame
            prepared_match = prepared_match[~prepared_match["formatted local time"].isin(result_timestamps)]

        # Due to merging with two tags at same timestamp, some rows are present twice, drop duplicates
        prepared_match.drop_duplicates(subset=["formatted local time", "full name", "x in m", "speed in m/s"], inplace=True)

        prepared_match["game"] = game

        if game == "FLEvsKIE":
            prepared_match.to_csv(f"handball_sample/MD05_Flensburg_Kiel/flensburg_kiel_md5_{file_name_add}.csv", sep=";")
        elif game == "ERLvsFLE":
            prepared_match.to_csv(f"handball_sample/MD10_Erlangen_Flensburg/erlangen_flensburg_md10_{file_name_add}.csv", sep=";")
        elif game == "FLEvsEIS":
            prepared_match.to_csv(f"handball_sample/MD11_Flensburg_Eisenach/flensburg_eisenach_md11_{file_name_add}.csv", sep=";")
        # elif game == "FLEvsRNL":
        #     result_df.to_csv(f"handball_sample/MD13_Flensburg_RNL/flensburg_rnl_md13_{file_name_add}.csv", sep=";")
        elif game == "GUMvsFLE":
            prepared_match.to_csv(f"handball_sample/MD14_Gummersbach_Flensburg/gummersbach_flensburg_md14_{file_name_add}.csv", sep=";")
        elif game == "FLEvsMEL":
            prepared_match.to_csv(f"handball_sample/MD15_Flensburg_Melsungen/flensburg_melsungen_md15_{file_name_add}.csv", sep=";")
        print("Single DataFrame saved")

    elif preparation_type == "all":
        # Load data
        match_fle_kie = pd.read_csv(f"handball_sample/MD05_Flensburg_Kiel/flensburg_kiel_md5_{file_name_add}.csv", index_col=0, sep=";")
        match_erl_fle = pd.read_csv(f"handball_sample/MD10_Erlangen_Flensburg/erlangen_flensburg_md10_{file_name_add}.csv", index_col=0, sep=";")
        match_fle_eis = pd.read_csv(f"handball_sample/MD11_Flensburg_Eisenach/flensburg_eisenach_md11_{file_name_add}.csv", index_col=0, sep=";")
        #match_fle_rnl = pd.read_csv(f"handball_sample/MD13_Flensburg_RNL/flensburg_rnl_md13{file_name_add}.csv", index_col=0, sep=";")
        match_gum_fle = pd.read_csv(f"handball_sample/MD14_Gummersbach_Flensburg/gummersbach_flensburg_md14_{file_name_add}.csv", index_col=0, sep=";")
        match_fle_mel = pd.read_csv(f"handball_sample/MD15_Flensburg_Melsungen/flensburg_melsungen_md15_{file_name_add}.csv", index_col=0, sep=";")
        
        match_full_train = pd.concat([match_fle_kie, match_erl_fle, match_fle_eis, match_gum_fle]).reset_index()
        # Create combination of timestamps and corresponding games to make sure correct data is used in each sample for later models
        match_full_train["timestamp_game"] = match_full_train["formatted local time"] + "_" + match_full_train["game"] 
        match_full_train.to_csv(f"handball_sample/match_training_{file_name_add}.csv", sep=";")

        match_full_test = pd.concat([match_fle_mel])
        # Create combination of timestamps and corresponding games to make sure correct data is used in each sample for later models
        match_full_test["timestamp_game"] = match_full_test["formatted local time"] + "_" + match_full_test["game"]
        match_full_test.to_csv(f"handball_sample/match_test_{file_name_add}.csv", sep=";")

        print("Complete DataFrames saved")

if __name__ == "__main__":
    for stats in ["True", "False"]:
        for game in  ["FLEvsKIE", "ERLvsFLE", "FLEvsEIS", "GUMvsFLE", "FLEvsMEL"]:
            print("----- single", game, " -----")
            main(preparation_type="single", game=game, stats=stats)
    print("Stats all true")
    main("all", "None", "True")
    print("Stats all false")
    main("all", "None", "False")