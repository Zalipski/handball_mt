import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import random
import os
from ast import literal_eval
from tsai.all import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.cluster import KMeans
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from joblib import dump, load

tuning = sys.argv[1]

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

set_every_seed()

def df_prep(match_df):
    """ Prepares the data for training the models.
    
    Parameters:
    match_df -- the DataFrame containing the match data

    Returns:
    aligned_data -- the prepared DataFrame
    """

    ball_data = match_df[match_df["full name"].str.contains("ball")]
    player_data = match_df[~match_df["full name"].str.contains("ball")]

    aligned_data = player_data.merge(ball_data[["formatted local time", "x in m", "y in m", "speed in m/s", "acceleration in m/s2", "direction of movement in deg"]], 
                                    on="formatted local time", how="left", suffixes=("_player", "_ball"))
    
    aligned_data["difference distance"] = np.sqrt(
        (aligned_data["x in m_player"] - aligned_data["x in m_ball"]) ** 2 +
        (aligned_data["y in m_player"] - aligned_data["y in m_ball"]) ** 2
    )

    aligned_data["difference speed"] = aligned_data["speed in m/s_player"] - aligned_data["speed in m/s_ball"]

    aligned_data["difference acceleration"] = aligned_data["acceleration in m/s2_player"] - aligned_data["acceleration in m/s2_ball"]

    aligned_data["difference direction"] = round((aligned_data["direction of movement in deg_player"] - aligned_data["direction of movement in deg_ball"] + 180) % 360 - 180, 3)

    return aligned_data

def discretize_data(discretization_vars, binned_vars, aligned_data, discretization_params):
    """ Discretizes the chosen variables in the given DataFrame.
    
    Parameters:
    discretization_vars   -- the list of variables to be discretized
    binned_vars           -- the list of binned variable names
    aligned_data          -- the DataFrame to be prepared
    discretization params -- the DataFrame containing the optimal parameters for each variable

    Returns:
    discretized_data -- the discretized DataFrame
    """

    discretized_data = aligned_data.copy()

    for i in range(len(binned_vars)):
        binned_variable = binned_vars[i]
        discretized_variable = discretization_vars[i]
        # Retrieve best bin edges for variable
        bins = discretization_params.loc[discretized_variable]["Best bins"]
        # Replace -inf inf to be able to use literal_eval to retrieve list without error
        bins = bins.replace("inf", "99999")
        bins = bins.replace("-inf", "-99999")
        bins = literal_eval(bins)
        
        technique = discretization_params.loc[discretized_variable]["Best technique"]
        if technique == "equal_width" or technique == "equal_freq":
            bins[-1] = np.inf # Replace "99999" as last value with np.inf
            bins[0] = -np.inf # Replace "-99999" as first value with -np.inf
            discretized_data[binned_variable] = pd.cut(discretized_data[discretized_variable], bins=bins, labels=range(1, len(bins)), include_lowest=True)
        elif technique == "k_means":
            # Reshape data for KMeans
            X_test = np.array(discretized_data[discretized_variable]).reshape(-1, 1)
            # Load best KMeans model for variable and use model to discretize it
            k_means = load(f"handball_sample\k_means_model_{binned_variable}.joblib")
            discretized_data = discretized_data.copy()
            discretized_data[binned_variable] = k_means.predict(X_test)

    return discretized_data

def model_tuning(train_df, val_df, tune_variable):
    """ Finds the best parameters based on the performance on the validation set.
    
    Parameters:
    train_df      -- the DataFrame containing the training data
    val_df        -- the DataFrame containing the validation data
    tune_variable -- the variable for which the best parameters should be found

    Returns:
    best_technique -- the best discretization technique
    best_bins      -- the best bin edges
    best_f1_score  -- the F1-score for the best parameters
    best_k_means   -- the best KMeans model
    """

    best_technique = "equal_width"
    best_bins = []
    best_f1_score = 0
    best_k_means = None
    delta = 0.05

    for number_of_bins in tqdm(range(2, 21)):
        for tuning_technique in ["equal_width", "equal_freq", "k_means"]:
            if tuning_technique == "equal_width":
                # Calculate bin edges for equal width bins
                bins = np.linspace(train_df[tune_variable].min(), train_df[tune_variable].max(), number_of_bins + 1)
                # Set first bin value to -infinity and last to infinity to ensure all values of unseen data are included
                bins[-1] = np.inf
                bins[0] = -np.inf
                bins = bins.tolist()
                # Discretize data
                train_df["variable_binned"] = pd.cut(train_df[tune_variable], bins=bins, labels=range(1, len(bins)), include_lowest=True)
                val_df["variable_binned"] = pd.cut(val_df[tune_variable], bins=bins, labels=range(1, len(bins)), include_lowest=True)
            elif tuning_technique == "equal_freq":
                # Discretize data
                train_df["variable_binned"], bins = pd.qcut(train_df[tune_variable], q=number_of_bins, labels=range(1, len(bins)), retbins=True)
                # Set first bin value to -inf and last to inf to ensure all values of unseen data are included
                bins[-1] = np.inf
                bins[0] = -np.inf
                bins = bins.tolist()
                # Discretize data again after bin updates
                train_df["variable_binned"] = pd.cut(train_df["variable_binned"], bins=bins, labels=range(1, len(bins)), include_lowest=True)
                val_df["variable_binned"] = pd.cut(val_df["variable_binned"], bins=bins, labels=range(1, len(bins)), include_lowest=True)
            else:
                # Reshape data for KMeans
                X_train = np.array(train_df[tune_variable]).reshape(-1, 1)
                X_val = np.array(val_df[tune_variable]).reshape(-1, 1)
                # Apply KMeans clustering
                k_means = KMeans(n_clusters=number_of_bins, random_state=42).fit(X_train)
                train_df["variable_binned"] = k_means.predict(X_train)
                val_df["variable_binned"] = k_means.predict(X_val)
                bins = []

            bn_model_features = [("variable_binned", "possession")]

            training_vars = ["variable_binned", "possession"]

            model = BayesianNetwork(bn_model_features)
            model.fit(train_df[training_vars], estimator=MaximumLikelihoodEstimator)

            # Evaluate on vaidation set
            test_f1, _, _, _, _ = evaluate_model(model, val_df[training_vars])
            if test_f1 > best_f1_score + delta:
                # If current F1-score is better than best F1-score + delta, update values
                best_technique = tuning_technique
                best_bins = bins
                best_f1_score = test_f1
                if best_technique == "k_means":
                    best_k_means = k_means
    
    return best_technique, best_bins, best_f1_score, best_k_means

def evaluate_model(model, data):
    """ Calculates the performance scores based on the used model and data.
    
    Parameters:
    model -- the trained model
    data  -- the data used for the model training

    Returns:
    f1            -- the F1-score
    precision     -- the Precision score
    recall        -- the Recall score
    predictions   -- the predictions
    probabilities -- the probability values of the predictions
    """
    
    infer = VariableElimination(model)
    predictions = []
    probabilities = []
    for _, row in data.iterrows():
        observed_evidence = row.drop("possession").to_dict()
        prediction = infer.query(variables=["possession"], evidence=observed_evidence)
        
        predicted_state = prediction.state_names["possession"][np.argmax(prediction.values)]
        predictions.append(predicted_state)
        
        prob_dist = prediction.values
        probabilities.append(prob_dist)
    true_values = data["possession"].tolist()
    f1 = f1_score(true_values, predictions, average="macro")
    precision = precision_score(true_values, predictions, average="macro")
    recall = recall_score(true_values, predictions, average="macro")
    return f1, precision, recall, predictions, probabilities

match_train_val_df = pd.read_csv(r"handball_sample\match_training_model.csv", sep=";", index_col=0)
match_test_df = pd.read_csv(r"handball_sample\match_test_model.csv", sep=";", index_col=0)

match_train_val_df["formatted local time"] = pd.to_datetime(match_train_val_df["formatted local time"])
match_test_df["formatted local time"] = pd.to_datetime(match_test_df["formatted local time"])

match_train_df = match_train_val_df[match_train_val_df["game"].isin(["FLEvsKIE", "ERLvsFLE", "FLEvsEIS"])]
match_val_df = match_train_val_df[match_train_val_df["game"].isin(["GUMvsFLE"])]

# Only use visible timestamps
visible_df_train = match_train_df[(match_train_df["tag text"] != "not_visible")]
visible_df_val = match_val_df[(match_val_df["tag text"] != "not_visible")]
visible_df_test = match_test_df[(match_test_df["tag text"] != "not_visible")]

# Prepare DataFrames
train_df_prep = df_prep(visible_df_train)
val_df_prep = df_prep(visible_df_val)
test_df_prep = df_prep(visible_df_test)

training_vars = ["difference distance", "x in m_player", "x in m_ball", "y in m_player", "y in m_ball", "speed in m/s_ball", "speed in m/s_player",
                 "acceleration in m/s2_ball", "acceleration in m/s2_player",
                 "direction of movement in deg_ball", "direction of movement in deg_player",
                 "difference speed", "difference acceleration", "difference direction"]
binned_training_vars = ["difference distance", "x_player_binned", "x_ball_binned", "y_player_binned", "y_ball_binned", "speed_ball_binned", "speed_player_binned",
                        "acceleration_ball_binned", "acceleration_player_binned",
                        "direction_ball_binned", "direction_player_binned",
                        "difference speed_binned", "difference acceleration_binned", "difference direction_binned"]

if tuning == "True":
    start_time_tuning = datetime.now()
    tuning_values = {}
    for i in tqdm(range(len(training_vars))):
        variable = training_vars[i]
        best_technique, best_number_of_bins, best_f1_score, best_k_means = model_tuning(train_df_prep, val_df_prep, variable)
        print("----")
        print(variable, best_technique, best_number_of_bins, best_f1_score)
        tuning_values[variable] = [best_technique, best_number_of_bins, best_f1_score]
        if best_technique == "k_means":
            # Save KMeans model if it is best technique
            dump(best_k_means, f"handball_sample\k_means_model_{binned_training_vars[i]}.joblib")
    tuning_df = pd.DataFrame(tuning_values).T
    tuning_df.columns = ["Best technique", "Best bins", "Best F1-score"]
    tuning_df.to_csv(r"handball_sample/best_params_bn.csv")
    end_time_tuning = datetime.now()
    print(end_time_tuning, "Time taken for tuning: ", end_time_tuning-start_time_tuning)
else:
    start_time_final = datetime.now()
    best_params = pd.read_csv(r"handball_sample\best_params_bn.csv", index_col=0)

    train_df_discr = discretize_data(training_vars, binned_training_vars, train_df_prep, best_params)
    test_df_discr = discretize_data(training_vars, binned_training_vars, test_df_prep, best_params)

    bn_model_features = [("difference distance", "possession"),
                        ("x_ball_binned", "possession"),
                        ("x_player_binned", "possession"),
                        ("y_ball_binned", "possession"),
                        ("y_player_binned", "possession"),
                        ("speed_ball_binned", "possession"),
                        ("speed_player_binned", "possession"),
                        ("acceleration_ball_binned", "possession"),
                        ("acceleration_player_binned", "possession"),
                        ("direction_ball_binned", "possession"),
                        ("direction_player_binned", "possession"),
                        ("difference speed_binned", "possession"),
                        ("difference acceleration_binned", "possession"),
                        ("difference direction_binned", "possession")]

    binned_training_vars += ["possession"]

    model = BayesianNetwork(bn_model_features)
    model.fit(train_df_discr[binned_training_vars], estimator=MaximumLikelihoodEstimator)

    # Evaluate on test set
    test_f1, test_precision, test_recall, test_predictions, test_probabilities = evaluate_model(model, test_df_discr[binned_training_vars])
    print("Test F1 Score:", test_f1)
    print("Test Precision:", test_precision)
    print("Test Recall:", test_recall)

    print(classification_report(test_df_discr["possession"], test_predictions))

    probability_of_possession = [arr[1] for arr in test_probabilities]

    test_df_discr["possession_pred_prob"] = probability_of_possession

    # Identify highest probability for possession within each timestamp
    test_df_discr["max_flag"] = test_df_discr.groupby("formatted local time", observed=True)["possession_pred_prob"].transform(lambda x: (x == x.max()).astype(int))

    test_df_discr["correct"] = (test_df_discr["max_flag"] == test_df_discr["possession"]).astype(int)

    total_timestamps = test_df_discr["formatted local time"].nunique()

    num_timestamps_all_correct = test_df_discr.groupby("formatted local time", observed=True)["correct"].all().sum()
    total_timestamps = test_df_discr["formatted local time"].nunique()
    print(num_timestamps_all_correct, total_timestamps, num_timestamps_all_correct / total_timestamps)
    end_time_final = datetime.now()
    print(end_time_final, "Time taken for final model: ", end_time_final-start_time_final)