import argparse
import os
import pandas as pd
import numpy as np
from scipy import stats
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold

def create_meta_df(file_path):
    print("\nGenerating Table with dataset info for diagnostic...")
    files = [file for file in os.listdir(file_path) if (file.endswith('.csv') and not 'test' in file and not 'train' in file)]
    meta_df = pd.DataFrame(columns=["file", "region", "interval", "units", "first_obs", "last_obs", "nans"])

    for i in range(len(files)):
        df_aux = pd.read_csv(os.path.join(file_path, files[i]))

        # Expresión regular para extraer la fecha y hora
        regex = r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})'

        # Aplicar la expresión regular para extraer la parte relevante y convertirla a fecha y hora
        df_aux['StartTime'] = df_aux['StartTime'].str.extract(regex)
        df_aux['StartTime'] = pd.to_datetime(df_aux['StartTime'], format='%Y-%m-%dT%H:%M')

        interval = df_aux['StartTime'].diff().mode().dt.total_seconds().values[0] / 3600  # Convertir a horas

        units = df_aux.UnitName[0] if len(np.unique(df_aux.UnitName)) == 1 else "Varying"

        first_obs = df_aux.StartTime[0]

        last_obs = df_aux.StartTime[len(df_aux.StartTime)-1]

        value_col = "quantity" if files[i].startswith('gen') else "Load"
        nans = df_aux[value_col].isnull().sum()

        new_row = [files[i], files[i].split("_")[1][:2], interval, units, first_obs, last_obs, nans]

        meta_df.loc[len(meta_df)] = new_row
    print("\nDATASET DIAGNOSTIC TABLE:")
    print(meta_df)
    return meta_df

def eval_meta_df(meta_df):
    warnings = []

    # Verificar si hay diferentes valores en la columna 'units'
    if meta_df['units'].nunique() > 1:
        warnings.append("\nWARNING: There are differing units in the data")
    if meta_df['interval'].nunique() > 1:
        warnings.append("\nWARNING: There are differing time intervals in the data")
    if meta_df['first_obs'].nunique() > 1:
        warnings.append("\nWARNING: Data doesn't start on the same date across all datasets")
    if meta_df['first_obs'].nunique() > 1:
        warnings.append("\nWARNING: Data doesn't end on the same date across all datasets")
    if len(warnings) < 1:
        warnings.append("\nNo inconsistencies were found in the data")
    for warn in warnings:
        print(warn)
        
def interpolate_and_sum_hour(hour_data):
    # Si todos los valores de la hora son NaN, devuelve NaN
    if hour_data.isnull().all():
        return np.nan
    
    # Si solo algunos valores son NaN, realiza la interpolación y suma
    if hour_data.notnull().any():
        interpolated_values = hour_data.interpolate()
        return interpolated_values.sum()
    
    # Si hay al menos un valor no NaN, pero no todos, devuelve NaN
    return np.nan

        
def llenar_horas_faltantes(df, region):
    #Guardamos tipo de energia (if necessary)
    isgen = True if "PsrType" in list(df.columns) else False
    if isgen:
        energy_type = df["PsrType"][0]
    
    # Expresión regular para extraer la fecha y hora
    regex = r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})'

    # Aplicar la expresión regular para extraer la parte relevante y convertirla a fecha y hora
    df['StartTime'] = df['StartTime'].str.extract(regex)
    df['StartTime'] = pd.to_datetime(df['StartTime'], format='%Y-%m-%dT%H:%M')
    
    # Calculamos la periodicidad en el dataset
    periodicidad = df['StartTime'].diff().mode().dt.total_seconds().values[0] / 3600
    
    #First we set start and end time
    start = pd.to_datetime("2022-01-01 00:00:00")
    end = pd.to_datetime("2023-01-01 00:00:00")
    
    # Agrupar por 'StartTime' y sumar los valores en 'value_col'
    value_col = df.columns[-1]
    df = df.groupby('StartTime')[value_col].sum().reset_index()
    
    # Establecer 'StartTime' como índice para trabajar con series de tiempo
    df = df.set_index('StartTime')

    # Crear un rango de fechas y horas esperadas con la periodicidad dada
    rango_fechas = pd.date_range(start=start, end=end, freq=f'{periodicidad}H')

    # Reindexar el DataFrame con el rango de fechas y horas esperadas
    df = df.reindex(rango_fechas).iloc[:-1]
    
    df = df.resample('H').apply(interpolate_and_sum_hour)

    # Rellenar valores faltantes en la columna 'quantity' usando interpolación
    #df[value_col] = df[value_col].interpolate()

    # Resamplear para tener un valor por hora y sumar los valores correspondientes
    #df = pd.DataFrame(df[value_col].resample('H').sum())
    
    #Renombrar columna
    if isgen:
        df = df.rename(columns={df.columns[0]: f"{region}_{value_col}_{energy_type}"})
    else:
        df = df.rename(columns={df.columns[0]: f"{region}_{value_col}"})

    return df

def create_basic_train_df(file_path):
    print("\nStarting preprocessing of the data:")
    print("\nGenerating basic dataset ...")
    train = pd.DataFrame()
    files = [file for file in os.listdir(file_path) if (file.endswith('.csv') and not 'test' in file and not 'train' in file)]
    for i in range(len(files)):
        df_aux = pd.read_csv(os.path.join(file_path, files[i]))
        region = files[i].split("_")[1][:2]
        df_aux = llenar_horas_faltantes(df_aux, region)
        train = pd.concat([train, df_aux], axis = 1)
    # Obtener todas las columnas que contienen "quantity"
    columns_to_sum = [col for col in train.columns if 'quantity' in col]

    # Extraer el código de país de estas columnas
    country_codes = list(set(col.split('_')[0] for col in columns_to_sum))
    
    # Crear las columnas de 'green_energy' para cada código de país y sumar las 'quantity' correspondientes
    for code in country_codes:
        quantity_columns = [col for col in columns_to_sum if col.startswith(f'{code}_')]
        train[f'{code}_green_energy'] = train[quantity_columns].sum(axis=1)
    
    # Eliminar las columnas 'quantity' que ya han sido sumadas
    train.drop(columns=columns_to_sum, inplace=True)
    print("DONE")
    return train

def handle_nans(df):
    original_length = len(df)
    df.replace(0, np.nan, inplace=True)
    df.dropna(how='any', inplace=True)
    final_length = len(df)
    
    percentage_dropped = (original_length - final_length) / original_length * 100
    
    if percentage_dropped > 10:
        print(f"\nWARNING: More than 10% of the data has been dropped. Percent dropped: {percentage_dropped:.2f}%")
    
    return df

def handle_outliers(df):
    for column in df.columns:
        outliers = np.abs(stats.zscore(df[column]))
        outlier_percentage = np.sum(outliers > 3) / len(outliers)  # Conteo de valores con z-score > 3
        
        if outlier_percentage < 0.02:
            continue  # Menos del 2% de outliers, no hacer nada
            
        elif 0.02 <= outlier_percentage <= 0.1:
            # Reemplazar outliers con valores interpolados en ambas direcciones
            df[column] = df[column].mask(outliers > 3).interpolate(limit_direction='both')
            
        else:
            # Superan el 10%, reemplazar con valores interpolados y mostrar un warning
            df[column] = df[column].mask(outliers > 3).interpolate(limit_direction='both')
            print(f"\nWARNING: {column} has more than 10% outliers.")
    
    return df
    
def add_surplus(train):
    print("\nAdding target variable...")
    country_codes = list(set(col.split('_')[0] for col in train.columns if '_' in col))

    # Iterar sobre los identificadores de países y calcular el 'surplus'
    for code in country_codes:
        green_energy_col = f'{code}_green_energy'
        load_col = f'{code}_Load'
        surplus_col = f'{code}_surplus'

        # Calcular el 'surplus' restando 'green_energy' - 'Load'
        train[surplus_col] = train[green_energy_col] - train[load_col]
    return train

def add_target(train):
    train['target'] = train.filter(like='_surplus').idxmax(axis=1).str[:2]
    
    # We need to shift the target column to reflect the reality of the predictions
    train['target'] = train['target'].shift(periods=-1)
    print("DONE")

    return train

def feature_eng(df):
    numeric_columns = [col for col in df.columns if col != 'target' and np.issubdtype(df[col].dtype, np.number)]

    # Generar los lags para las columnas numéricas
    lags = [1, 2, 3, 24]
    lag_columns = []
    for col in numeric_columns:
        for lag in lags:
            lag_columns.append(df[col].shift(lag).rename(f"{col}_lag_{lag}"))

    target_lag_columns = [df["target"].shift(lag).rename(f"target_lag_{lag}") for lag in lags]

    df = pd.concat([df] + lag_columns + target_lag_columns, axis=1)

    # Calcula la media acumulativa por día para cada columna numérica
    daily_means = df[numeric_columns].groupby(df.index.date).expanding().mean().reset_index(level=0, drop=True)

    # Calcula la media acumulativa por mes para cada columna numérica
    monthly_means = df[numeric_columns].groupby(df.index.to_period('M')).expanding().mean().reset_index(level=0, drop=True)

    # Une las medias diarias y mensuales al DataFrame original
    df = df.join(daily_means.add_suffix('_daily_mean')).join(monthly_means.add_suffix('_monthly_mean'))

    # Feature extraction de la variable tiempo
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['hour'] = df.index.hour

    # Definir las columnas y sus correspondientes funciones trigonométricas
    time_columns = [('hour', 24), ('day', 30), ('month', 12)]

    # Iterar a través de las columnas y aplicar las transformaciones trigonométricas
    for column, period in time_columns:
        df[f'sin_{column}'] = np.sin(2 * np.pi * df[column] / period)
        df[f'cos_{column}'] = np.cos(2 * np.pi * df[column] / period)

    # Eliminar las columnas originales si es necesario
    df = df.drop(columns=[column for column, _ in time_columns])
    
    # Mapear las columnas de target
    mapping_dict = {
        'SP': 0, 'UK': 1, 'DE': 2, 'DK': 3, 'HU': 5,
        'SE': 4, 'IT': 6, 'PO': 7, 'NL': 8
    }

    columns_to_map = [col for col in df.columns if "target" in col]
    for col in columns_to_map:
        df[col] = df[col].replace(mapping_dict)
        
    return df

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('\nMemory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        #else:
        #    df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def select_features(df):
    # Supongamos que tienes un DataFrame df con características predictoras y la columna 'target'
    # Separar las características y la columna objetivo
    X = df.drop('target', axis=1)
    y = df['target']

    # Número de veces para realizar la validación cruzada
    n_iterations = 5

    # Almacenar las importancias de características de cada iteración
    all_lgb_importances = []
    all_xgb_importances = []

    # Dividir los datos en conjunto de entrenamiento y prueba para la validación cruzada
    kf = KFold(n_splits=n_iterations, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # LightGBM
        lgb_model = lgb.LGBMRegressor(verbosity=-1)
        lgb_model.fit(X_train, y_train)
        lgb_feature_importance = pd.Series(lgb_model.feature_importances_, index=X.columns)

        # XGBoost
        xgb_model = xgb.XGBRegressor()
        xgb_model.fit(X_train, y_train)
        xgb_feature_importance = pd.Series(xgb_model.feature_importances_, index=X.columns)

        all_lgb_importances.append(lgb_feature_importance)
        all_xgb_importances.append(xgb_feature_importance)

    # Calcular la importancia media de las características de LightGBM y XGBoost
    mean_lgb_importance = pd.concat(all_lgb_importances, axis=1).mean(axis=1).sort_values(ascending=False)
    mean_xgb_importance = pd.concat(all_xgb_importances, axis=1).mean(axis=1).sort_values(ascending=False)

    # Seleccionar las características con mayor importancia
    num_features_to_select = 25  # Número de características a seleccionar
    intercalated_vars = [val for pair in zip(mean_lgb_importance.index.values, mean_xgb_importance.index.values) for val in pair]
    # Eliminar duplicados manteniendo el orden
    seen = set()
    selected_features = [x for x in intercalated_vars if not (x in seen or seen.add(x))][:25]
    selected_features.append("target")
    
    return selected_features

def save_data(train, output_file):
    train.to_csv(os.path.join(output_file, "train.csv"))
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Data processing script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--file_path',
        type=str,
        default='data/raw_data.csv',
        help='Path to the raw data file to process'
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        default='data/processed_data.csv', 
        help='Path to save the processed data'
    )
    return parser.parse_args()

def main(file_path, output_path):
    meta_df = create_meta_df(file_path)    # comment if not needed
    eval_meta_df(meta_df)                  # comment if not needed   (they're for diagnostic purposes)
    train = create_basic_train_df(file_path)   # creates basic training info with strictly necessary info
    train = handle_nans(train)      # adds surplus for each country and the target variable
    train = handle_outliers(train)
    train = add_surplus(train)
    train = add_target(train)
    train = feature_eng(train)
    train = reduce_mem_usage(train)
    train = train[select_features(train[:int(train.shape[0]*0.8)])]
    save_data(train, output_path)
    print("\nPreprocessing finished.")

if __name__ == "__main__":
    args = parse_arguments()
    main(args.file_path, args.output_path)