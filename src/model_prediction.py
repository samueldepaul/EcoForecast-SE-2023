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

def load_test_data(file_path):
    X_test = pd.read_csv(os.path.join(file_path, "X_test.csv")).drop(columns=['Unnamed: 0'])
    y_test = pd.read_csv(os.path.join(file_path, "y_test.csv"))["target"]
    return X_test, y_test

def load_labelencoder(file_path):
    label_encoder = joblib.load(os.path.join(file_path, 'label_encoder.pkl'))
    return label_encoder

def load_model(model_path):
    model = joblib.load(os.path.join(model_path, 'best_model.pkl'))
    return model

def make_baseline_predictions(X_test, label_encoder):
    baseline_prediction = label_encoder.transform(X_test["target_lag_1"])
    return baseline_prediction

def evaluate_baseline_prediction(baseline_prediction, y_test):
    f1 = f1_score(baseline_prediction, y_test, average='weighted')
    print("\nTHE BASELINE PREDICTION F1-SCORE ON THE TEST DATA IS: {}".format(f1))

def make_predictions(X_test, model):
    predictions = model.predict(X_test)
    return predictions

def evaluate_predictions(predictions, y_test):
    f1 = f1_score(predictions, y_test, average='weighted')
    print("\nTHE MODEL'S F1-SCORE ON THE TEST DATA IS: {}".format(f1))

def save_predictions(predictions, label_encoder, predictions_file):
    formatted_predictions = label_encoder.inverse_transform(predictions).astype(int).tolist()
    json_data = {"target":{str(indice): valor for indice, valor in enumerate(formatted_predictions)}}
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
    X_test, y_test = load_test_data(input_file)
    label_encoder = load_labelencoder(input_file)
    model = load_model(model_file)
    baseline_prediction = make_baseline_predictions(X_test, label_encoder)
    evaluate_baseline_prediction(baseline_prediction, y_test)
    predictions = make_predictions(X_test, model)
    evaluate_predictions(predictions, y_test)
    save_predictions(predictions, label_encoder, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file, args.output_file)
