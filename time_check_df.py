import pandas as pd
import numpy as np
from datetime import timedelta
import sys

game = sys.argv[1]

def round_to_nearest_50ms(dt):
    """ Round the timestamps to nearest 50ms.
    
    Parameters:
    dt -- the datetime timestamps

    Returns:
    rounded_dt -- the timestamps rounded to nearest 50ms
    """
    # Convert microseconds to milliseconds and round to nearest 50 milliseconds
    milliseconds = round(dt.microsecond / 1000.0 / 50.0) * 50
    # Rebuild datetime with new milliseconds, adjusting for overflow
    new_second = dt.second
    if milliseconds >= 1000:
        milliseconds -= 1000
        new_second += 1
    # Ensure seconds don't overflow
    if new_second >= 60:
        new_second -= 60
        dt += timedelta(minutes=1)
    rounded_dt = dt.replace(second=new_second, microsecond=int(milliseconds * 1000))
    return rounded_dt

if __name__ == "__main__":
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
    elif game == "FLEvsRNL":
        match_h1 = pd.read_csv(r"handball_sample\MD13_Flensburg_RNL\SG_Flensburg-Handewitt_vs._Rhe_phase_1._Halbzeit__positions.csv", sep=";")
        match_h2 = pd.read_csv(r"handball_sample\MD13_Flensburg_RNL\SG_Flensburg-Handewitt_vs._Rhe_phase_2._Halbzeit__positions.csv", sep=";")
        tags = pd.read_csv(r"handball_sample\MD13_Flensburg_RNL\tags_flensburg_rnl_md13_s2324.csv", sep=";")
    elif game == "GUMvsFLE":
        match_h1 = pd.read_csv(r"handball_sample\MD14_Gummersbach_Flensburg\VfL_Gummersbach_vs._SG_Flensbu_phase_1._Halbzeit_positions.csv", sep=";")
        match_h2 = pd.read_csv(r"handball_sample\MD14_Gummersbach_Flensburg\VfL_Gummersbach_vs._SG_Flensbu_phase_2._Halbzeit_positions.csv", sep=";")
        tags = pd.read_csv(r"handball_sample\MD14_Gummersbach_Flensburg\tags_gummersbach_flensburg_md14_s2324.csv", sep=";")
    elif game == "FLEvsMEL":
        match_h1 = pd.read_csv(r"handball_sample\MD15_Flensburg_Melsungen\SG_Flensburg-Handewitt_vs._MT__phase_1.HZ_positions.csv", sep=";")
        match_h2 = pd.read_csv(r"handball_sample\MD15_Flensburg_Melsungen\SG_Flensburg-Handewitt_vs._MT__phase_2.HZ_positions.csv", sep=";")
        tags = pd.read_csv(r"handball_sample\MD15_Flensburg_Melsungen\tags_flensburg_melsungen_md15_s2324.csv", sep=";")

    tags.drop(["player"], axis=1, inplace=True)

    match_full = pd.concat([match_h1, match_h2]).reset_index().drop(["index", "heart rate in bpm", "core temperature in celsius", "player orientation in deg",  "Unnamed: 23"], axis=1)
    match_full["formatted local time"] = pd.to_datetime(match_full["formatted local time"])

    match_start = match_full[match_full[~np.isnan(match_full["ball possession (id of possessed ball)"])].index[0]:]
    match_start["time diff from start"] = match_start["formatted local time"] - match_start["formatted local time"].iloc[0]

    match_start_poss = match_start.dropna(subset=["ball possession (id of possessed ball)"])
    # Drop consecutive rows where same player is still in possession, this is achieved by comparing each row with previous row and keeping only rows where player in possession changes
    match_start_poss = match_start_poss[match_start_poss["sensor id"] != match_start_poss["sensor id"].shift(1)]

    # Get game start time
    if game == "FLEvsEIS":
        table_start_time = pd.to_datetime("2023-10-28 19:02:14.000") # No player has possession in first half of that game according to Kinexon data
    # elif game == "FLEvsRNL":
    #     table_start_time = pd.to_datetime("2023-11-18 18:02:10.650") # Possession starts too late
    else:
        table_start_time = match_start_poss["formatted local time"].iloc[0]

    offset = 0

    tags["# time (in ms)"] = tags["# time (in ms)"] - tags["# time (in ms)"][tags["tag text"] == "game_start"].iloc[0]
    # Create adjusted timestamp column where video time of tags is converted to actual timestamp 
    tags["adjusted_timestamp"] = tags["# time (in ms)"].apply(lambda x: table_start_time + timedelta(milliseconds=x + offset))
    tags["rounded_timestamp"] = tags["adjusted_timestamp"].apply(round_to_nearest_50ms) # Round adjusted timestamps

    # Forward fill tags df to have correct merging of match and tags df 
    video_start_time = tags["rounded_timestamp"].iloc[0]
    table_end_time = match_full["formatted local time"].iloc[-1]

    # Create timestamp df with frequency of 20Hz (frequency of Kinexon data) from game start to end time
    all_timestamps = pd.date_range(start=video_start_time, end=table_end_time, freq="50L")
    all_timestamps_df = pd.DataFrame(all_timestamps, columns=["timestamp"])

    # Merge tags with all timestamps and forward fill
    tags_full = pd.merge(all_timestamps_df, tags, left_on="timestamp", right_on="rounded_timestamp", how="left")
    cols_from_tags = tags.columns.difference(["# time (in ms)", "adjusted_timestamp"])
    for col in cols_from_tags:
        tags_full[col] = tags_full[col].ffill()
        
    merged_df = pd.merge(match_full, tags_full, left_on="formatted local time", right_on="timestamp", how="left")

    checkpoints = tags[tags["tag text"] == "time_check_position"].shape[0] # Get number of checkpoints

    for tag_to_check in range(checkpoints):
        ts_to_check = tags[tags["tag text"].isin(["time_check_position"])]["rounded_timestamp"].iloc[tag_to_check]
        # Find index where first of i-th check-tag is used
        try:
            # Get previous and next timestamps based on timestamp to check for consideration
            check_min, check_max = match_full[match_full["formatted local time"] == ts_to_check].index[0] - 150, match_full[match_full["formatted local time"] == ts_to_check].index[0] + 165
        except IndexError:
            # If timestamp to check is not available in the data, proceed to next timestamp
            print(ts_to_check, "Data not available")
            continue

        df_to_check = match_full[match_full["group name"] == "Ball"].loc[check_min:check_max]
        if df_to_check.size == 0:
            print("Ball is not found in area to check")
            continue

        if df_to_check["z in m"].min() > 5:
            # If z value is over 5m in every considered timestamp, proceed to next timestamp
            print(ts_to_check, "Z sensor in ball wrong, this position can't be reliably checked")
            continue

        time_in_df = df_to_check["formatted local time"][df_to_check["z in m"] == df_to_check["z in m"].min()]

        # Print the difference between tabular and video times
        difference = abs(time_in_df - ts_to_check)
        print("Checkpoint", tag_to_check, ", Video", ts_to_check, ", Difference in ms is:", difference.iloc[0].microseconds/1000, ", Tabular", time_in_df.iloc[0])