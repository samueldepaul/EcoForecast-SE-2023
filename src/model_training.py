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


def load_data(file_path):
    df = pd.read_csv(os.path.join(file_path, "train.csv"))
    return df

def split_data(df, file_path):
    X = df.drop(columns=['Unnamed: 0']) 
    X = X.drop(columns=['target']) 
    y = df["target"]

    # Dividir los datos en un split 80-20 respetando el orden
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Crear una instancia de LabelEncoder
    label_encoder = LabelEncoder()

    # Aplicar label encoding a y_train
    y_train = label_encoder.fit_transform(y_train)
    y_test[:-1] = label_encoder.transform(y_test[:-1])
    
    # Guardar los archivos en la carpeta de datos
    X_train.to_csv(os.path.join(file_path, 'X_train.csv'))
    X_test[:-1].to_csv(os.path.join(file_path, 'X_test.csv'))
    pd.DataFrame(y_train, columns=['target']).to_csv(os.path.join(file_path, 'y_train.csv'))
    pd.DataFrame(y_test[:-1], columns=['target']).to_csv(os.path.join(file_path, 'y_test.csv'))
    
    joblib.dump(label_encoder, os.path.join(file_path, 'label_encoder.pkl'))
    
    return X_train, X_test[:-1], y_train, y_test


def train_model(X_train, y_train):
    # Definir diferentes combinaciones de hiperparámetros
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.01],
        'n_estimators': [10, 50]
    }

    results = []

    # Crear k-folds
    k_folds = 7
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    for model_name, Model in [('LightGBM', lgb.LGBMClassifier), ('XGBoost', xgb.XGBClassifier)]:
        for params in product(*param_grid.values()):
            param_dict = dict(zip(param_grid.keys(), params))
            if model_name == 'LightGBM':
                model = Model(**param_dict, verbosity=-1)  # Establecer verbosity=-1 para LightGBM
            else:
                model = Model(**param_dict)

            f1_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='f1_weighted')

            # Obtener la media de los F1-scores de validación cruzada
            f1_mean = f1_scores.mean()

            results.append((model_name, param_dict, f1_mean))

    # Crear un DataFrame con los resultados
    results_df = pd.DataFrame(results, columns=['Model', 'Parameters', 'Mean F1-score'])
    return results_df

def save_best_model(results_df, model_file, X_train, y_train):
    # Identificar el mejor modelo según el F1-score
    best_model_info = results_df.loc[results_df['Mean F1-score'].idxmax()]
    best_model_name = best_model_info['Model']
    best_model_params = best_model_info['Parameters']

    # Crear y entrenar el mejor modelo con todos los datos de entrenamiento
    best_model = None
    if best_model_name == 'LightGBM':
        best_model = lgb.LGBMClassifier(**best_model_params)
    elif best_model_name == 'XGBoost':
        best_model = xgb.XGBClassifier(**best_model_params)

    # Entrenar el modelo con todos los datos de entrenamiento
    best_model.fit(X_train, y_train)

    # Guardar el mejor modelo en un archivo .pkl
    joblib.dump(best_model, os.path.join(model_file, 'best_model.pkl'))
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Model training script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/', 
        help='Path to the processed data file to train the model'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='models/', 
        help='Path to save the trained model'
    )
    return parser.parse_args()

def main(input_file, model_file):
    df = load_data(input_file)
    X_train, X_val, y_train, y_val = split_data(df, input_file)
    model = train_model(X_train, y_train)
    save_best_model(model, model_file, X_train, y_train)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file)