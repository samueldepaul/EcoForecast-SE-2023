import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, GroupKFold, StratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error
from math import sqrt
import itertools
from sklearn.metrics import f1_score
import timeit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
import xgboost as xgb
from itertools import product
import joblib
import argparse
import json

import os
import pandas as pd
import joblib
from sklearn.metrics import f1_score
import json

def load_test_data(file_path):
    """
    Load test data from CSV files and return the X_test and y_test.

    Parameters:
    file_path (str): Path to the directory containing test data files.

    Returns:
    pd.DataFrame, pd.Series: X_test (features), y_test (target).
    """
    X_test = pd.read_csv(os.path.join(file_path, "X_test.csv")).drop(columns=['Unnamed: 0'])
    y_test = pd.read_csv(os.path.join(file_path, "y_test.csv"))["target"]
    return X_test, y_test

def load_labelencoder(file_path):
    """
    Load a label encoder from a file.

    Parameters:
    file_path (str): Path to the label encoder file.

    Returns:
    LabelEncoder: Loaded label encoder object.
    """
    label_encoder = joblib.load(os.path.join(file_path, 'label_encoder.pkl'))
    return label_encoder

def load_model(model_path):
    """
    Load a trained model from a file.

    Parameters:
    model_path (str): Path to the directory containing the model file.

    Returns:
    Model: Loaded model object.
    """
    model = joblib.load(os.path.join(model_path, 'best_model.pkl'))
    return model

def make_baseline_predictions(X_test, label_encoder):
    """
    Create baseline predictions using a label encoder.

    Parameters:
    X_test (pd.DataFrame): Test data.
    label_encoder (LabelEncoder): Loaded label encoder.

    Returns:
    np.array: Baseline predictions.
    """
    baseline_prediction = label_encoder.transform(X_test["target_lag_1"])
    return baseline_prediction

def evaluate_baseline_prediction(baseline_prediction, y_test):
    """
    Evaluate baseline predictions using F1-score.

    Parameters:
    baseline_prediction (np.array): Baseline predictions.
    y_test (pd.Series): True target labels.

    Returns:
    None
    """
    f1 = f1_score(baseline_prediction, y_test, average='weighted')
    print("\nTHE BASELINE PREDICTION F1-SCORE ON THE TEST DATA IS: {}".format(f1))

def make_predictions(X_test, model):
    """
    Make predictions using a trained model.

    Parameters:
    X_test (pd.DataFrame): Test data.
    model (Model): Trained model.

    Returns:
    np.array: Predictions.
    """
    predictions = model.predict(X_test)
    return predictions

def evaluate_predictions(predictions, y_test):
    """
    Evaluate predictions using F1-score.

    Parameters:
    predictions (np.array): Model predictions.
    y_test (pd.Series): True target labels.

    Returns:
    None
    """
    f1 = f1_score(predictions, y_test, average='weighted')
    print("\nTHE MODEL'S F1-SCORE ON THE TEST DATA IS: {}".format(f1))

def save_predictions(predictions, label_encoder, predictions_file):
    """
    Save predictions in a JSON file.

    Parameters:
    predictions (np.array): Model predictions.
    label_encoder (LabelEncoder): Loaded label encoder.
    predictions_file (str): Path to the directory to save predictions.

    Returns:
    None
    """
    formatted_predictions = label_encoder.inverse_transform(predictions).astype(int).tolist()
    json_data = {"target": {str(index): value for index, value in enumerate(formatted_predictions)}}
    with open(os.path.join(predictions_file, 'predictions.json'), 'w') as file:
        json.dump(json_data, file)
    pass


def parse_arguments():
    parser = argparse.ArgumentParser(description='Prediction script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/test_data.csv', 
        help='Path to the test data file to make predictions'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='models/model.pkl',
        help='Path to the trained model file'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='predictions/predictions.json', 
        help='Path to save the predictions'
    )
    return parser.parse_args()

def main(input_file, model_file, output_file):
    # Load test data for evaluation
    X_test, y_test = load_test_data(input_file)
    
    # Load label encoder
    label_encoder = load_labelencoder(input_file)
    
    # Load trained model
    model = load_model(model_file)
    
    # Create baseline predictions using the label encoder
    baseline_prediction = make_baseline_predictions(X_test, label_encoder)
    
    # Evaluate baseline predictions
    evaluate_baseline_prediction(baseline_prediction, y_test)
    
    # Make predictions using the trained model
    predictions = make_predictions(X_test, model)
    
    # Evaluate model predictions
    evaluate_predictions(predictions, y_test)
    
    # Save predictions to an output file with label encoder for decoding
    save_predictions(predictions, label_encoder, output_file)


if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file, args.output_file)
