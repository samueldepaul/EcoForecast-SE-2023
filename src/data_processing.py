import argparse
import os
import pandas as pd
import numpy as np
from scipy import stats
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold



def create_meta_df(file_path):
    """
    Create a metadata DataFrame containing information about datasets.

    Args:
    - file_path (str): Path to the directory containing CSV files.

    Returns:
    - meta_df (pd.DataFrame): DataFrame with dataset information.
    """
    print("\nGenerating Table with dataset info for diagnostic...")
    # Filter files by criteria and create a list of relevant files
    files = [file for file in os.listdir(file_path) if file.endswith('.csv') and 'test' not in file and 'train' not in file]
    
    # Define columns for metadata DataFrame
    columns = ["file", "region", "interval", "units", "first_obs", "last_obs", "nans"]
    meta_df = pd.DataFrame(columns=columns)

    for file in files:
        file_path_full = os.path.join(file_path, file)
        df_aux = pd.read_csv(file_path_full)

        # Regular expression to extract date and time
        regex = r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})'
        # Extract relevant part and convert to date and time
        df_aux['StartTime'] = pd.to_datetime(df_aux['StartTime'].str.extract(regex)[0], format='%Y-%m-%dT%H:%M')

        # Calculate interval in hours
        interval = df_aux['StartTime'].diff().mode().dt.total_seconds().values[0] / 3600

        # Determine units
        units = df_aux.UnitName.iloc[0] if len(df_aux['UnitName'].unique()) == 1 else "Varying"

        # Extract first and last observation timestamps
        first_obs = df_aux['StartTime'].iloc[0]
        last_obs = df_aux['StartTime'].iloc[-1]

        # Define the column based on the file name and count NaNs
        value_col = "quantity" if file.startswith('gen') else "Load"
        nans = df_aux[value_col].isnull().sum()

        # Create a new row with metadata information
        new_row = [file, file.split("_")[1][:2], interval, units, first_obs, last_obs, nans]

        # Append the new row to the metadata DataFrame
        meta_df.loc[len(meta_df)] = new_row

    # Display dataset diagnostic table
    print("\nDATASET DIAGNOSTIC TABLE:")
    print(meta_df)
    return meta_df



def check_data_consistency(meta_df):
    """
    Check for inconsistencies in the given DataFrame columns and print warnings if found.

    Args:
    meta_df (pandas.DataFrame): DataFrame containing metadata.

    Returns:
    None
    """

    warnings = []

    # Check for differing units in the 'units' column
    if meta_df['units'].nunique() > 1:
        warnings.append("WARNING: There are differing units in the data")

    # Check for differing time intervals in the 'interval' column
    if meta_df['interval'].nunique() > 1:
        warnings.append("WARNING: There are differing time intervals in the data")

    # Check if data doesn't start on the same date across all datasets
    if meta_df['first_obs'].nunique() > 1:
        warnings.append("WARNING: Data doesn't start on the same date across all datasets")

    # Check if data doesn't end on the same date across all datasets
    if meta_df['last_obs'].nunique() > 1:
        warnings.append("WARNING: Data doesn't end on the same date across all datasets")

    # If no inconsistencies were found, add a message indicating that
    if not warnings:
        warnings.append("No inconsistencies were found in the data")

    # Print the warnings
    for warning in warnings:
        print(warning)
        
        
        
def interpolate_and_sum_hour(hour_data):
    """
    Interpolates missing values in the input hour_data (if any) and returns the sum of the values.
    
    Args:
    - hour_data (pd.Series): Input data for a particular hour
    
    Returns:
    - float or NaN: Sum of the values after interpolation or NaN if all values are NaN
    
    Raises:
    - ValueError: If the input is not a pandas Series
    """
    # Check if hour_data is a pandas Series
    if not isinstance(hour_data, pd.Series):
        raise ValueError("Input hour_data must be a pandas Series")
    
    # If all values of the hour are NaN, return NaN
    if hour_data.isnull().all():
        return np.nan
    
    # If some values are NaN, perform interpolation and return the sum
    if hour_data.notnull().any():
        interpolated_values = hour_data.interpolate()
        return interpolated_values.sum()
    
    # If at least one value is not NaN but not all, return NaN
    return np.nan


        
def fill_missing_hours(df, region):
    """
    Fills missing hours in a DataFrame of time series data and aggregates values per hour.

    Args:
    - df (DataFrame): Input DataFrame containing time series data.
    - region (str): Name of the region for column naming.

    Returns:
    - DataFrame: Modified DataFrame with missing hours filled and aggregated values per hour.
    """
    is_energy_type_present = "PsrType" in df.columns
    if is_energy_type_present:
        energy_type = df["PsrType"][0]
    
    # Define regex to extract date and time
    regex_datetime = r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})'

    # Extract relevant date and time, converting it to datetime format
    df['StartTime'] = df['StartTime'].str.extract(regex_datetime)
    df['StartTime'] = pd.to_datetime(df['StartTime'], format='%Y-%m-%dT%H:%M')
    
    # Calculate dataset periodicity
    periodicity = df['StartTime'].diff().mode().dt.total_seconds().values[0] / 3600
    
    # Set start and end time
    start_time = pd.to_datetime("2022-01-01 00:00:00")
    end_time = pd.to_datetime("2023-01-01 00:00:00")
    
    # Group by 'StartTime' and sum values in 'value_col'
    value_col = df.columns[-1]
    df = df.groupby('StartTime')[value_col].sum().reset_index()
    
    # Set 'StartTime' as index for time series operations
    df = df.set_index('StartTime')

    # Create a range of expected dates and hours based on the given periodicity
    date_range = pd.date_range(start=start_time, end=end_time, freq=f'{periodicity}H')

    # Reindex the DataFrame with the expected date and hour range, excluding the last entry
    df = df.reindex(date_range).iloc[:-1]
    
    # Apply an operation for filling missing values and summing values per hour
    df = df.resample('H').apply(interpolate_and_sum_hour)

    # Rename columns based on region and energy type presence
    if is_energy_type_present:
        df = df.rename(columns={df.columns[0]: f"{region}_{value_col}_{energy_type}"})
    else:
        df = df.rename(columns={df.columns[0]: f"{region}_{value_col}"})

    return df



def create_basic_train_df(file_path):
    """
    Creates a basic training DataFrame by concatenating data from CSV files in the given directory.
    
    Args:
    - file_path (str): The path to the directory containing CSV files.

    Returns:
    - pandas.DataFrame: The concatenated DataFrame.
    """
    print("\nStarting preprocessing of the data:")
    print("\nGenerating basic dataset ...")
    
    # Get relevant CSV files for training
    files = [file for file in os.listdir(file_path) if (file.endswith('.csv') and 'test' not in file and 'train' not in file)]
    
    # Initialize an empty DataFrame to store concatenated data
    train = pd.DataFrame()

    # Concatenate data from each CSV file into the train DataFrame
    for file in files:
        df_aux = pd.read_csv(os.path.join(file_path, file))
        region = file.split("_")[1][:2]
        
        # Fill missing hours in the data for the given region
        df_aux = fill_missing_hours(df_aux, region)
        
        # Concatenate the data horizontally
        train = pd.concat([train, df_aux], axis=1)
    
    # Get columns containing 'quantity'
    columns_to_sum = [col for col in train.columns if 'quantity' in col]
    
    # Extract country codes from column names
    country_codes = list(set(col.split('_')[0] for col in columns_to_sum))
    
    # Create 'green_energy' columns for each country code and sum corresponding 'quantity' columns
    for code in country_codes:
        quantity_columns = [col for col in columns_to_sum if col.startswith(f'{code}_')]
        train[f'{code}_green_energy'] = train[quantity_columns].sum(axis=1)
    
    # Remove the 'quantity' columns that have been summed
    train.drop(columns=columns_to_sum, inplace=True)
    
    print("DONE")
    return train



def handle_nans(df):
    """
    Handles NaN values in a DataFrame by replacing 0s with NaNs and dropping rows with any NaNs.
    
    Args:
    - df (pd.DataFrame): Input DataFrame
    
    Returns:
    - pd.DataFrame: DataFrame with NaN values handled
    """
    original_length = len(df)
    df.replace(0, np.nan, inplace=True)
    df.dropna(how='any', inplace=True)
    final_length = len(df)
    
    percentage_dropped = (original_length - final_length) / original_length * 100
    
    if percentage_dropped > 10:
        print(f"\nWARNING: More than 10% of the data has been dropped. Percent dropped: {percentage_dropped:.2f}%")
    
    return df



def handle_outliers(df):
    """
    Handles outliers in a DataFrame using z-score method for each column.
    Outliers are identified and replaced with interpolated values based on the percentage of outliers.
    
    Args:
    - df (pd.DataFrame): Input DataFrame
    
    Returns:
    - pd.DataFrame: DataFrame with outliers handled
    """
    for column in df.columns:
        z_scores = np.abs(stats.zscore(df[column]))
        outlier_percentage = np.sum(z_scores > 3) / len(z_scores)  # Percentage of values with z-score > 3
        
        if outlier_percentage < 0.02:
            continue  # Less than 2% outliers, do nothing
            
        elif 0.02 <= outlier_percentage <= 0.1:
            # Replace outliers with interpolated values in both directions
            df[column] = df[column].mask(z_scores > 3).interpolate(limit_direction='both')
            
        else:
            # More than 10% outliers, replace with interpolated values and display a warning
            df[column] = df[column].mask(z_scores > 3).interpolate(limit_direction='both')
            print(f"\nWARNING: {column} has more than 10% outliers.")
    
    return df
    
def add_surplus(train):
    """
    Calculate surplus for each country based on green energy and load columns.

    Args:
    - train (DataFrame): Input DataFrame containing columns related to energy data.

    Returns:
    - train (DataFrame): DataFrame with surplus columns added for each country.
    """
    print("\nCalculating surplus...")
    
    # Extract unique country codes from column names
    country_codes = {col.split('_')[0] for col in train.columns if '_' in col}

    # Calculate surplus for each country
    for code in country_codes:
        green_energy_col = f'{code}_green_energy'
        load_col = f'{code}_Load'
        surplus_col = f'{code}_surplus'

        # Calculate surplus by subtracting load from green energy
        train[surplus_col] = train[green_energy_col] - train[load_col]
    
    return train



def add_target(train):
    """
    Add target column based on surplus values and shift it for predictions.

    Args:
    - train (DataFrame): Input DataFrame containing surplus data.

    Returns:
    - train (DataFrame): DataFrame with 'target' column added and shifted.
    """
    train['target'] = train.filter(like='_surplus').idxmax(axis=1).str[:2]
    
    # Shift the target column to reflect prediction reality
    train['target'] = train['target'].shift(periods=-1)
    
    print("Target column added and shifted.")
    return train



def feature_eng(df):
    """
    Perform feature engineering on the input DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing target and numeric columns.

    Returns:
    pd.DataFrame: Processed DataFrame with engineered features.
    """

    # Identify numeric columns excluding the target column
    numeric_columns = [col for col in df.columns if col != 'target' and np.issubdtype(df[col].dtype, np.number)]

    # Generate lagged columns for numeric features
    lags = [1, 2, 3, 24]
    lag_columns = [df[col].shift(lag).rename(f"{col}_lag_{lag}") for col in numeric_columns for lag in lags]

    # Create lagged columns for the target variable
    target_lag_columns = [df["target"].shift(lag).rename(f"target_lag_{lag}") for lag in lags]

    # Concatenate lagged columns with the original DataFrame
    df = pd.concat([df] + lag_columns + target_lag_columns, axis=1)

    # Calculate daily and monthly cumulative means for numeric columns
    daily_means = df[numeric_columns].groupby(df.index.date).expanding().mean().reset_index(level=0, drop=True)
    monthly_means = df[numeric_columns].groupby(df.index.to_period('M')).expanding().mean().reset_index(level=0, drop=True)

    # Merge daily and monthly means with the original DataFrame
    df = df.join(daily_means.add_suffix('_daily_mean')).join(monthly_means.add_suffix('_monthly_mean'))

    # Extract temporal features: month, day, hour
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['hour'] = df.index.hour

    # Define columns and corresponding trigonometric functions
    time_columns = [('hour', 24), ('day', 30), ('month', 12)]

    # Apply trigonometric transformations to temporal columns
    for column, period in time_columns:
        df[f'sin_{column}'] = np.sin(2 * np.pi * df[column] / period)
        df[f'cos_{column}'] = np.cos(2 * np.pi * df[column] / period)

    # Drop original temporal columns
    df = df.drop(columns=[column for column, _ in time_columns])

    # Map target columns according to predefined dictionary
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
    """
    Selects the most important features using LightGBM and XGBoost.

    Parameters:
    df (pd.DataFrame): DataFrame containing predictor features and the 'target' column.

    Returns:
    list: List of selected features including the 'target'.
    """

    # Separate predictor features and the target column
    X = df.drop('target', axis=1)
    y = df['target']

    # Number of iterations for cross-validation
    n_iterations = 5

    # Store feature importances for each iteration
    all_lgb_importances = []
    all_xgb_importances = []

    # Split data into training and test sets for cross-validation
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

    # Calculate mean feature importance for LightGBM and XGBoost
    mean_lgb_importance = pd.concat(all_lgb_importances, axis=1).mean(axis=1).sort_values(ascending=False)
    mean_xgb_importance = pd.concat(all_xgb_importances, axis=1).mean(axis=1).sort_values(ascending=False)

    # Select features with highest importance
    num_features_to_select = 25  # Number of features to select
    intercalated_vars = [val for pair in zip(mean_lgb_importance.index.values, mean_xgb_importance.index.values) for val in pair]
    # Remove duplicates while preserving order
    seen = set()
    selected_features = [x for x in intercalated_vars if not (x in seen or seen.add(x))][:num_features_to_select]
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
    # Load metadata if required
    meta_df = create_meta_df(file_path)  # Commented out if metadata not needed
    check_data_consistency(meta_df)  # Commented out if not required for diagnostics

    # Create a basic training dataset with essential information
    train = create_basic_train_df(file_path)

    # Handling missing values and outliers
    train = handle_nans(train)  # Add surplus for each country and the target variable
    train = handle_outliers(train)
    
    # Feature engineering
    train = add_surplus(train)
    train = add_target(train)
    train = feature_eng(train)
    
    # Optimize memory usage
    train = reduce_mem_usage(train)
    
    # Select important features and trim the dataset
    train = train[select_features(train[:int(train.shape[0] * 0.8)])]
    
    # Save processed data
    save_data(train, output_path)
    
    print("\nPreprocessing finished.")

if __name__ == "__main__":
    args = parse_arguments()
    main(args.file_path, args.output_path)