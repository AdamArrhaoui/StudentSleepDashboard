import pandas as pd
import numpy as np
import os
from scipy import stats

RAW_DATA_PATH = "../data/raw/student_sleep_screen_time_academic_performance.csv"

def load_data_file(path=RAW_DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist!")
    data = pd.read_csv(path)
    return data

def inspect_data(df: pd.DataFrame):
    print(f"Data loaded:")
    print(df.head())
    print(f"\nData shape {df.shape}")
    print(f"\nColumn types:\n{df.dtypes}")
    print("\nDescription:")
    print(df.describe(include="all"))
    print("\nMissing values:")
    print(df.isna().sum())
    print(f"\nNumber of rows with missing values: {df.isna().any(axis=1).sum()}")
    df = handle_missing(df)
    print(f"\nDataset size after handling missing values: {df.shape[0]}")
    df = remove_outliers(df)
    print(f"\nDataset size after removing outliers: {df.shape[0]}")

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    # We just drop all rows with missing values.
    # Only 44/500 rows had missing values, so it's better to just drop them
    # than to try to extrapolate missing data which could lead to inaccuracy
    return df.dropna()

def remove_outliers(df: pd.DataFrame, z_score_threshold: float = 3.0) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include="number").columns
    keep_mask = pd.Series(True, index=df.index)

    for col in numeric_cols:
        zscores = np.abs(stats.zscore(df[col], nan_policy="omit"))
        zscores = pd.Series(zscores, index=df.index)  # realign to df's index

        is_outlier = zscores >= z_score_threshold
        keep_mask &= ~is_outlier
    return df[keep_mask]

if __name__ == '__main__':
    df = load_data_file()
    inspect_data(df)