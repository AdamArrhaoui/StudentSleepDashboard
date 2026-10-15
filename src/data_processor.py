from typing import Self

import pandas as pd
import numpy as np
import os
from scipy import stats

RAW_DATA_PATH = "../data/raw/student_sleep_screen_time_academic_performance.csv"

class DataProcessor:
    """Loads, processes, and saves a dataset to be used for linear regression"""
    data: pd.DataFrame

    def __init__(self, path=RAW_DATA_PATH):
        self.load_data_file(path)

    def load_data_file(self, path=RAW_DATA_PATH) -> Self:
        if not os.path.exists(path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(script_dir, path)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Path {path} does not exist!")
        self.data = pd.read_csv(path)
        return self

    def inspect_data(self) -> Self:
        df = self.data
        print(f"Data loaded:")
        print(df.head())
        print(f"\nData shape {df.shape}")
        print(f"\nColumn types:\n{df.dtypes}")
        print("\nDescription:")
        print(df.describe(include="all"))
        print("\nMissing values:")
        print(df.isna().sum())
        print(f"\nNumber of rows with missing values: {df.isna().any(axis=1).sum()}")
        #df = handle_missing(df)
        #print(f"\nDataset size after handling missing values: {df.shape[0]}")
        #df = remove_outliers(df)
        #print(f"\nDataset size after removing outliers: {df.shape[0]}")
        return self

    def handle_missing(self) -> Self:
        # We just drop all rows with missing values.
        # Only 44/500 rows had missing values, so it's better to just drop them
        # than to try to extrapolate missing data which could lead to inaccuracy
        self.data.dropna(inplace=True)
        return self

    def remove_outliers(self, z_score_threshold: float = 3.0) -> Self:
        df = self.data
        numeric_cols = df.select_dtypes(include="number").columns
        keep_mask = pd.Series(True, index=df.index)

        for col in numeric_cols:
            zscores = np.abs(stats.zscore(df[col], nan_policy="omit"))
            zscores = pd.Series(zscores, index=df.index)  # realign to df's index

            is_outlier = zscores >= z_score_threshold
            keep_mask &= ~is_outlier
        self.data = df[keep_mask]
        return self

if __name__ == '__main__':
    processor = DataProcessor()
    processor.inspect_data()