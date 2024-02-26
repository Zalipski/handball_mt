import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import random
import os
from tsai.all import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

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

    aligned_data = player_data.merge(ball_data[['formatted local time', 'x in m', 'y in m', 'speed in m/s']], 
                                    on='formatted local time', how='left', suffixes=("_player", "_ball"))
    
    aligned_data['distance_to_ball'] = np.sqrt(
        (aligned_data['x in m_player'] - aligned_data['x in m_ball']) ** 2 +
        (aligned_data['y in m_player'] - aligned_data['y in m_ball']) ** 2
    )

    bins = [0.2, 0.5, 1, 5, 10, 50, np.inf]
    labels = [1, 2, 3, 4, 5, 6]
    aligned_data['distance_to_ball_binned'] = pd.cut(aligned_data['distance_to_ball'], bins=bins, labels=labels)
    aligned_data['speed_player_binned'] = pd.cut(aligned_data['speed in m/s_player'], bins=bins, labels=labels)
    aligned_data['speed_ball_binned'] = pd.cut(aligned_data['speed in m/s_ball'], bins=bins, labels=labels)

    aligned_data.dropna(subset=["distance_to_ball_binned", "possession", "speed_ball_binned", "speed_player_binned"], inplace=True)
    return aligned_data

def evaluate_model(model, data, target_variable='possession'):
    """ Calculates the performance scores based on the used model and data.
    
    Parameters:
    model            -- the trained model
    data             -- the data used for the model training
    target_variable  -- the target variable to be predicted

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
        observed_evidence = row.drop(target_variable).to_dict()
        prediction = infer.query(variables=[target_variable], evidence=observed_evidence)
        
        predicted_state = prediction.state_names[target_variable][np.argmax(prediction.values)]
        predictions.append(predicted_state)
        
        prob_dist = prediction.values
        probabilities.append(prob_dist)
    true_values = data[target_variable].tolist()
    f1 = f1_score(true_values, predictions, average='macro')
    precision = precision_score(true_values, predictions, average='macro')
    recall = recall_score(true_values, predictions, average='macro')
    return f1, precision, recall, predictions, probabilities

match_train_df = pd.read_csv(r"handball_sample\match_training.csv", sep=";", index_col=0)
match_test_df = pd.read_csv(r"handball_sample\match_test.csv", sep=";", index_col=0)

match_train_df['formatted local time'] = pd.to_datetime(match_train_df['formatted local time'])
match_test_df['formatted local time'] = pd.to_datetime(match_test_df['formatted local time'])

# Only use visible timestamps for training and testing
visible_df_train = match_train_df[(match_train_df["tag text"] != "not_visible")]
visible_df_test = match_test_df[(match_test_df["tag text"] != "not_visible")]

train_df, val_df = train_test_split(visible_df_train, test_size=0.2, random_state=42)

train_df = df_prep(train_df)
val_df = df_prep(val_df)
test_df = df_prep(visible_df_test)

bn_model_features = [('distance_to_ball_binned', 'possession'),
                    ('speed_ball_binned', 'possession'),
                    ('speed_player_binned', 'possession')]

training_data = ["distance_to_ball_binned", "possession", "speed_ball_binned", "speed_player_binned"]

model = BayesianNetwork(bn_model_features)
model.fit(train_df[training_data], estimator=MaximumLikelihoodEstimator)

# Evaluate on validation set
val_f1, val_precision, val_recall, _, _ = evaluate_model(model, val_df[training_data])
print("Validation F1 Score:", val_f1)
print("Validation Precision:", val_precision)
print("Validation Recall:", val_recall)

# Evaluate on test set
test_f1, test_precision, test_recall, test_predictions, test_probabilities = evaluate_model(model, test_df[training_data])
print("Test F1 Score:", test_f1)
print("Test Precision:", test_precision)
print("Test Recall:", test_recall)

print(classification_report(test_df['possession'], test_predictions))

probability_of_possession = [arr[1] for arr in test_probabilities]

test_df["possession_pred_prob"] = probability_of_possession

# Identify highest probability for possession within each timestamp
test_df['max_flag'] = test_df.groupby('formatted local time')['possession_pred_prob'].transform(lambda x: (x == x.max()).astype(int))

test_df['correct'] = (test_df['max_flag'] == test_df['possession']).astype(int)

total_timestamps = test_df['formatted local time'].nunique()

num_timestamps_all_correct = test_df.groupby('formatted local time')['correct'].all().sum()
total_timestamps = test_df['formatted local time'].nunique()
print(num_timestamps_all_correct, total_timestamps, num_timestamps_all_correct / total_timestamps)