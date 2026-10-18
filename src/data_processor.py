from typing import Self

import pandas as pd
import numpy as np
import os
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

RAW_DATA_PATH = "../data/raw/student_sleep_screen_time_academic_performance.csv"
PROCESSED_DATA_PATH = "../data/processed/"

# --------------------------------------------------------------------------------------
# Column config
# --------------------------------------------------------------------------------------
TARGET = "cgpa"

# we don't need the student id column, and gender column should be avoided to remove a potential source of bias
DROP_COLUMNS = ["student_id", "gender"]

# columns without a pre-defined minimum and/or maximum -> StandardScaler (mean/std learned from the training set)
STANDARD_COLUMNS = [
    "age",
    "year_of_study",
    "weekend_sleep_delta_hours",
    "pre_sleep_screen_time_min",
    "caffeine_drinks_per_day",
    "naps_per_week",
    "weekly_study_hours",
]

# bedtime is a clock hour, so a bedtime after midnight is stored as e.g. 0.5, which lands at the opposite end of the
# scale from 23.5 even though the two are an hour apart. Hours before BEDTIME_DAY_START are shifted past midnight
# (+24) so the clock reads continuously: 23.5 -> 23.5, 0.5 -> 24.5.
BEDTIME_COLUMN = "avg_bedtime_hour"
BEDTIME_DAY_START = 12

# columns scaled with MinMaxScaler (min/max learned from the training set). bedtime is unwrapped first (see above)
MINMAX_COLUMNS = [BEDTIME_COLUMN]

# columns with strict, known ranges (hours in a day, days in a week, 0-10 ratings, etc)
FIXED_RANGES = {
    "avg_sleep_hours_weekday": (0, 24),
    "daily_screen_time_hours": (0, 24),
    "sleep_quality_score": (0, 10),
    "exercise_days_per_week": (0, 7),
    "energy_level": (0, 10),
    "concentration_difficulty": (0, 10),
    "class_attendance_percent": (0, 100),
}

# categorical columns are one-hot encoded
CATEGORICAL_COLUMNS = [
    "major_category",
    "residence_type",
    "chronotype",
    "primary_device",
    "device_free_before_bed",
    "has_part_time_job",
    "semester_type",
]

NUMERIC_COLUMNS = STANDARD_COLUMNS + MINMAX_COLUMNS + list(FIXED_RANGES)
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

def _resolve_path(path: str) -> str:
    """Use the path as given if it exists, otherwise treat it as relative to this script's folder."""
    if os.path.exists(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, path)

class DataProcessor:
    """Loads, processes, and saves a dataset to be used for linear regression"""
    data: pd.DataFrame

    def __init__(self, path=RAW_DATA_PATH):
        self.load_data_file(path)

    def load_data_file(self, path: str) -> Self:
        path = _resolve_path(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist!")
        self.data = pd.read_csv(path)
        return self

    def split_and_save_data(self, save_folder_path: str, train_frac: float = 0.70,
                val_frac: float = 0.15,
                test_frac: float = 0.15,
                random_state: int = 42) -> Self:
        # make sure fractions add up to 1
        assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9, "Fractions must sum to 1"
        # ensure save folder paths
        save_folder_path = _resolve_path(save_folder_path)
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        # split off the training set. everything else goes into a temporary holdout
        train_df, holdout_df = train_test_split(
            self.data, train_size=train_frac, random_state=random_state, shuffle=True
        )
        # split the holdout into validation and test
        # (the test share of the holdout is test_frac / (val_frac + test_frac), i.e. 0.5 for 15/15)
        test_share = test_frac / (val_frac + test_frac)
        val_df, test_df = train_test_split(
            holdout_df, test_size=test_share, random_state=random_state
        )
        # save datasets
        train_df.to_csv(os.path.join(save_folder_path, "train.csv"), index=False)
        val_df.to_csv(os.path.join(save_folder_path, "validation.csv"), index=False)
        test_df.to_csv(os.path.join(save_folder_path, "test.csv"), index=False)
        print(f"Datasets have been saved to {os.path.abspath(save_folder_path)}")
        return self

    def inspect_data(self, df: pd.DataFrame | None = None) -> Self:
        pd.set_option('display.max_columns', None)
        df = self.data if df is None else df
        print(f"Data loaded:")
        print(df.head())
        print(f"\nData shape {df.shape}")
        print(f"\nColumn types:\n{df.dtypes}")
        print("\nDescription:")
        print(df.describe(include="all"))
        print("\nMissing values:")
        print(df.isna().sum())
        print(f"\nNumber of rows with missing values: {df.isna().any(axis=1).sum()}")
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

    def drop_columns(self, columns: list[str]) -> Self:
        self.data.drop(columns, axis=1, inplace=True)
        return self

    def remove_duplicates(self) -> Self:
        self.data.drop_duplicates(inplace=True)
        return self

    def scale_minmax(self, columns: list[str]) -> Self:
        scaler = MinMaxScaler()
        self.data[columns] = scaler.fit_transform(self.data[columns])
        return self

    def scale_standard(self, columns: list[str]) -> Self:
        scaler = StandardScaler()
        self.data[columns] = scaler.fit_transform(self.data[columns])
        return self

    def onehot_encode(self, columns: list[str]) -> Self:
        self.data = pd.get_dummies(self.data, columns=columns, dtype=int)
        return self

    @staticmethod
    def _scale_fixed_ranges(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, (low, high) in FIXED_RANGES.items():
            df[col] = (df[col] - low) / (high - low)
        return df

def run_processing_pipeline(input_file=RAW_DATA_PATH, output_folder=PROCESSED_DATA_PATH, verbose: bool = False):
     processor = DataProcessor(input_file)
     if verbose:
        print("###### PRE-PROCESSED DATA ######")
        processor.inspect_data()
     # we don't need the student id column, and gender column should be avoided to remove a potential source of bias
     processor.drop_columns(DROP_COLUMNS)
     # now that student_id is gone, we can remove the duplicate values
     processor.remove_duplicates()
     # we drop rows with missing values because there are not that many of them
     processor.handle_missing()
     # we remove columns with extreme outliers values that are over 3 standard deviations away from the mean value
     processor.remove_outliers(3.0)
     # we apply standard scaler to columns that don't have a pre-defined minimum and/or maximum value
     processor.scale_standard(STANDARD_COLUMNS)
     # we apply minmax scaler to columns with a well-defined min and max
     processor.scale_minmax(MINMAX_COLUMNS)
     # we apply custom minmax scaling ranges to columns with strict ranges (hours in a day/days in a week, 0-10 ratings, cgpa, etc)
     processor._scale_fixed_ranges()
     # we apply one-hot encoding to categorical attributes
     processor.onehot_encode(CATEGORICAL_COLUMNS)
     if verbose:
        print("######   PROCESSED DATA   ######")
        processor.inspect_data()
     processor.split_and_save_data(output_folder)

if __name__ == '__main__':
    run_processing_pipeline(RAW_DATA_PATH, PROCESSED_DATA_PATH, verbose=True)

