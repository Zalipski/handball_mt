import numpy as np
import pandas as pd
from datetime import timedelta
from tqdm import tqdm
import random
import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from tsai.all import *
import optuna
from optuna.integration import FastAIPruningCallback

window_length_ms, include_tuning = int(sys.argv[1]), str(sys.argv[2])

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

def pivot_df_prep(match_df):
    """ Uses the pivot function to have all rows per timestamp condensed into one row.
    
    Parameters:
    match_df -- the DataFrame containing the match data

    Returns:
    pivot_df -- the match DataFrame after the pivot transformation 
    """

    # Separate DataFrame for ball and players to correctly create numbers for players, otherwise number of ball row index is skipped
    ball_df = match_df[match_df["full name"].str.contains("ball", case=False)].copy()
    players_df = match_df[~match_df["full name"].str.contains("ball", case=False)].copy()

    # Assign player numbers, starting from 1
    players_df["Player"] = players_df.groupby("formatted local time").cumcount() + 1
    players_df["Player"] = players_df.apply(lambda x: "Ball" if "ball" in x["full name"].lower() else f"Player{x['Player']}", axis=1)
    ball_df["Player"] = "Ball"

    combined_df = pd.concat([ball_df, players_df])

    pivot_df = combined_df.pivot_table(index="formatted local time", columns="Player", values=["x in m", "y in m", "z in m", "speed in m/s",
                                                                                               "direction of movement in deg", "acceleration in m/s2"])

    # Flatten multi-index columns
    pivot_df.columns = ["_".join(col).strip() for col in pivot_df.columns.values]

    # Extract unique player names and sort them
    players = sorted(set([col.split("_")[1] for col in pivot_df.columns if "Player" in col]), key=lambda x: int(x.replace("Player", "")))

    # Reorder columns to group XYZ for each player
    ordered_columns = ["x in m_Ball", "y in m_Ball", "z in m_Ball", "speed in m/s_Ball", "direction of movement in deg_Ball",
                    "acceleration in m/s2_Ball"]
    for player in players:
        ordered_columns.extend([f"x in m_{player}", f"y in m_{player}", f"z in m_{player}", f"speed in m/s_{player}", f"direction of movement in deg_{player}",
                    f"acceleration in m/s2_{player}"])

    pivot_df = pivot_df[ordered_columns]
    pivot_df.reset_index(inplace=True)

    pivot_df["formatted local time"] = pd.to_datetime(pivot_df["formatted local time"])
    return pivot_df

def sample_retrieval(pivot_df, label_df, window_start, window_end):
    """ Retrieves the needed data for each sample by filtering out the window of the chosen pivot DataFrame.
    
    Parameters:
    pivot_df        -- the DataFrame the pivot function should be performed on
    label_df        -- the regular match data containing the possession labels for each timestamp
    window_start    -- the start of the window
    window_end      -- the end of the window

    Returns:
    df_timestamp     -- the DataFrame containing all chosen input variables
    possession_label -- the label stating who is in possession
    usable_sample    -- states if the data for the chosen time window has enough timesteps
    """

    usable_sample = "True"
    df_timestamp = pivot_df[(pivot_df["formatted local time"] >= window_start) & 
                            (pivot_df["formatted local time"] <= window_end)]
    df_len = len(df_timestamp)
    if df_len < input_time_steps:
        # Discard sample if less than needed amount of input timesteps are present
        usable_sample = "False"
        return None, None, usable_sample

    # Get data of considered timestamp
    df_timestamp = df_timestamp[["x in m_Ball", "y in m_Ball", "z in m_Ball", "speed in m/s_Ball", "direction of movement in deg_Ball", "acceleration in m/s2_Ball",
                                "x in m_Player1", "y in m_Player1", "z in m_Player1", "speed in m/s_Player1", "direction of movement in deg_Player1", "acceleration in m/s2_Player1",
                                "x in m_Player2", "y in m_Player2", "z in m_Player2", "speed in m/s_Player2", "direction of movement in deg_Player2", "acceleration in m/s2_Player2",
                                "x in m_Player3", "y in m_Player3", "z in m_Player3", "speed in m/s_Player3", "direction of movement in deg_Player3", "acceleration in m/s2_Player3",
                                "x in m_Player4", "y in m_Player4", "z in m_Player4", "speed in m/s_Player4", "direction of movement in deg_Player4", "acceleration in m/s2_Player4",
                                "x in m_Player5", "y in m_Player5", "z in m_Player5", "speed in m/s_Player5", "direction of movement in deg_Player5", "acceleration in m/s2_Player5",
                                "x in m_Player6", "y in m_Player6", "z in m_Player6", "speed in m/s_Player6", "direction of movement in deg_Player6", "acceleration in m/s2_Player6",
                                "x in m_Player7", "y in m_Player7", "z in m_Player7", "speed in m/s_Player7", "direction of movement in deg_Player7", "acceleration in m/s2_Player7",
                                "x in m_Player8", "y in m_Player8", "z in m_Player8", "speed in m/s_Player8", "direction of movement in deg_Player8", "acceleration in m/s2_Player8",
                                "x in m_Player9", "y in m_Player9", "z in m_Player9", "speed in m/s_Player9", "direction of movement in deg_Player9", "acceleration in m/s2_Player9",
                                "x in m_Player10", "y in m_Player10", "z in m_Player10", "speed in m/s_Player10", "direction of movement in deg_Player10", "acceleration in m/s2_Player10",
                                "x in m_Player11", "y in m_Player11", "z in m_Player11", "speed in m/s_Player11", "direction of movement in deg_Player11", "acceleration in m/s2_Player11",
                                "x in m_Player12", "y in m_Player12", "z in m_Player12", "speed in m/s_Player12", "direction of movement in deg_Player12", "acceleration in m/s2_Player12",
                                "x in m_Player13", "y in m_Player13", "z in m_Player13", "speed in m/s_Player13", "direction of movement in deg_Player13", "acceleration in m/s2_Player13",
                                "x in m_Player14", "y in m_Player14", "z in m_Player14", "speed in m/s_Player14", "direction of movement in deg_Player14", "acceleration in m/s2_Player14"]].to_numpy()
    
    # Get possession label by checking which row (i.e. player) is in possession at final timestamp
    label = label_df[(label_df["formatted local time"] == window_end)]["possession"].to_numpy()
    label = np.argmax(label)
    return df_timestamp, label, usable_sample

def create_samples(timestamp_game, data_amount, pivot_df, match_df):
    """ Creates the complete samples for the train/val/test set.
    
    Parameters:
    timestamp_game -- the DataFrame the pivot function should be performed on
    data_amount    -- the amount of samples to be considered
    pivot_df       -- the pivot transformed DataFrame
    match_df       -- the end of the window

    Returns:
    tsd -- the TimeseriesDataset containing the X and y tensors
    """

    timestamps = pd.to_datetime([tg.split("_")[0] for tg in timestamp_game]) # Get timestamps from timestamp_game combination for window calculations
    window_starts = timestamps - window_length_datetime # Pre-calculate start and end times for each window

    X_list = []
    y_list = []

    for window_start, window_end in tqdm(zip(window_starts[:data_amount], timestamps[:data_amount])):
        sample, label, usable_sample = sample_retrieval(pivot_df, match_df, window_start, window_end) # Get data for every timestamp sample by calling sample_retrieval function
        if usable_sample == "False":
            continue # Skip sample if it is not usable
        X_list.append(sample)
        y_list.append(label)

    # Convert lists to numpy arrays
    X = np.array(X_list)
    y = np.array(y_list)

    X_na = np.nan_to_num(X, nan=-99) # Replace NaN values with -99
    X_na = X_na.transpose(0, 2, 1) # Transpose to have correct ordering for TST model 

    # Convert the data to PyTorch tensors
    X_tensor = torch.tensor(X_na, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    print(X_tensor.shape, y_tensor.shape)

    # Create TSDatasets
    tsd = TSDatasets(X_tensor, y_tensor)

    return tsd

def objective(trial:optuna.Trial):
    """ Creates the search space for hyperparameter tuning and executes the tuning.
    
    Parameters:
    trial -- the current Optuna trial object

    Returns:
    best_valid_loss -- the best validation loss value
    """

    number_of_epochs = 50
    early_stopping = True # Boolean stating wether early stopping should be used during tuning

    # Search space
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    dropout = trial.suggest_float("dropout", 0.0, 0.3, step=0.1)
    fc_dropout = trial.suggest_float("fc_dropout", 0.0, 0.8, step=0.1)
    n_layers = trial.suggest_categorical("n_layers", [2, 3, 4, 5, 6, 7, 8])
    d_model = trial.suggest_categorical("d_model", [128, 256, 512, 1024])
    n_heads = trial.suggest_categorical("n_heads", [2, 8, 10, 12, 14, 16])
    d_k = trial.suggest_categorical("d_k", [None, 16, 32, 64, 128, 256, 512])
    d_v = trial.suggest_categorical("d_v", [None, 16, 32, 64, 128, 256, 512])
    d_ff = trial.suggest_categorical("d_ff", [256, 512, 1024, 2048, 4096])

    dls = TSDataLoaders.from_dsets(train_ds, val_ds, bs=batch_size, tfms = [None, TSClassification()], batch_tfms=TSStandardize(), num_workers=0)
    model = TST(c_in=dls.vars, c_out=15, seq_len=dls.len, n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=dropout, fc_dropout=fc_dropout)
    learn = Learner(dls, model, loss_func=nn.CrossEntropyLoss(), metrics=[accuracy], cbs=FastAIPruningCallback(trial, monitor="valid_loss"))
    
    if early_stopping == False:
        # with ContextManagers([learn.no_logging(), learn.no_bar()]): # Prevents printing anything during training
        learn.fit_one_cycle(number_of_epochs, lr_max=learning_rate)
    else:
        best_valid_loss = float("inf")
        epochs_since_improvement = 0
        patience = 10
        delta = 0.05

        for _ in range(number_of_epochs):
            #with ContextManagers([learn.no_logging(), learn.no_bar()]):
            learn.fit_one_cycle(1, lr_max = learning_rate)
            current_valid_loss = learn.recorder.values[-1][1]

            if best_valid_loss - current_valid_loss > delta:
                best_valid_loss = current_valid_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= patience:
                raise optuna.exceptions.TrialPruned()  # Prune trial if patience is exceeded and not enough improvement has been made

    return best_valid_loss

# Load data
match_train_df = pd.read_csv(r"handball_sample\match_training.csv", sep=";", index_col=0)
match_test_df = pd.read_csv(r"handball_sample\match_test.csv", sep=";", index_col=0)

match_train_df["formatted local time"] = pd.to_datetime(match_train_df["formatted local time"])
match_test_df["formatted local time"] = pd.to_datetime(match_test_df["formatted local time"])

# Only use visible timestamps
visible_df_train = match_train_df[(match_train_df["tag text"] != "not_visible")]
visible_df_test = match_test_df[(match_test_df["tag text"] != "not_visible")]

# Get unique combinations of timestamp and game
timestamp_game_combinations_train = visible_df_train["timestamp_game"].unique()
test_timestamp_game = visible_df_test["timestamp_game"].unique()

train_timestamp_game, val_timestamp_game = train_test_split(timestamp_game_combinations_train, test_size=0.2, random_state=42)

# Change DataFrame structure
print("Pivoting DataFrames")
pivot_df_train = pivot_df_prep(match_train_df)
pivot_df_test = pivot_df_prep(match_test_df)

# Prepare train, val and test data
print("Pivoting DataFrames")
train_ds = create_samples(timestamp_game=train_timestamp_game, data_amount=10000, pivot_df=pivot_df_train, match_df=match_train_df)
val_ds = create_samples(timestamp_game=val_timestamp_game, data_amount=1000, pivot_df=pivot_df_train, match_df=match_train_df)
test_ds = create_samples(timestamp_game=test_timestamp_game, data_amount=1000, pivot_df=pivot_df_test, match_df=match_test_df)

if include_tuning == "True":
    print("Starting Optuna study")
    study = optuna.create_study(direction="minimize") # Minimize validation loss
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
    params_df = pd.DataFrame(trial.params, index = [0])
    params_df.to_csv(r"handball_sample\best_params_tst.csv")

# Load best params
params_tst = pd.read_csv(r"handball_sample\best_params_tst.csv", index_col = 0)

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

# Final model with best params
print("Training final model")
dls = TSDataLoaders.from_dsets(train_ds, val_ds, bs=batch_size, tfms = [None, TSClassification()], batch_tfms=TSStandardize(), num_workers=0)
model = TST(c_in=dls.vars, c_out=15, seq_len=dls.len, n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=dropout, fc_dropout=fc_dropout)
learn = Learner(dls, model, loss_func=nn.CrossEntropyLoss(), metrics=[accuracy], cbs=None)
learn.fit_one_cycle(50, lr_max=learning_rate)

learn.export("handball_sample/tst_model.pth") # Save final model
learner_test = load_learner("handball_sample/tst_model.pth", cpu=False)

# Create a DataLoader from the test dataset
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# Evaluate performance on test data
loss, accuracy = learner_test.validate(dl=test_dl)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

preds, y_true = learner_test.get_preds(dl=test_dl)

# Convert predictions to class indices
class_preds = torch.argmax(preds, dim=1)

print("Performance of final model on test data:")
accuracy = (class_preds == y_true).float().mean()
print(f"Accuracy: {accuracy.item()}")