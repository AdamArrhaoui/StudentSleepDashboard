# Python 3.10 compatibility: typing.Self only exists in Python 3.11 and newer.
# Our conda environment and the Docker image (python:3.10-slim) use Python 3.10,
# so we fall back to the typing_extensions package, which provides the same Self.
try:
    from typing import Self  # Python 3.11+
except ImportError:
    from typing_extensions import Self  # Python 3.10 (our environment and Docker image)

import joblib
import pandas as pd
import numpy as np
import os
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.utils.validation import check_is_fitted

RAW_DATA_PATH = "../data/raw/student_sleep_screen_time_academic_performance.csv"
PROCESSED_DATA_PATH = "../data/processed/"
PREPROCESSOR_FILENAME = "preprocessor.joblib"

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
    preprocessor: ColumnTransformer
    splits: dict[str, pd.DataFrame]

    def __init__(self, path: str | None = RAW_DATA_PATH):
        # path=None skips loading raw data, for when you only want to transform rows with a saved preprocessor
        self.data = pd.DataFrame()
        self.splits = {}
        self.preprocessor = self._build_preprocessor()  # unfitted until split_and_save_data / load_preprocessor
        if path is not None:
            self.load_data_file(path)

    # ----------------------------------------------------------------------------------
    # Loading and inspecting
    # ----------------------------------------------------------------------------------
    def load_data_file(self, path: str) -> Self:
        path = _resolve_path(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist!")
        self.data = pd.read_csv(path)
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

    # ----------------------------------------------------------------------------------
    # Row-level cleaning (whole dataset only, never applied to a single prediction row)
    # ----------------------------------------------------------------------------------
    def drop_columns(self, columns: list[str]) -> Self:
        self.data.drop(columns, axis=1, inplace=True)
        return self

    def remove_duplicates(self) -> Self:
        self.data.drop_duplicates(inplace=True)
        return self

    def handle_missing(self) -> Self:
        # We just drop all rows with missing values.
        # Only 44/500 rows had missing values, so it's better to just drop them
        # than to try to extrapolate missing data which could lead to inaccuracy
        self.data.dropna(inplace=True)
        return self

    def unwrap_bedtime(self) -> Self:
        # Run this BEFORE remove_outliers
        self.data = self._unwrap_bedtime(self.data)
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
        self.data = df[keep_mask].copy()
        return self

    # ----------------------------------------------------------------------------------
    # Scaling and encoding
    # ----------------------------------------------------------------------------------
    @staticmethod
    def _build_preprocessor() -> ColumnTransformer:
        """Builds the unfitted ColumnTransformer that scales and encodes the feature columns."""
        preprocessor = ColumnTransformer(
            transformers=[
                ("standard", StandardScaler(), STANDARD_COLUMNS),
                ("minmax", MinMaxScaler(), MINMAX_COLUMNS),
                # these were already scaled by _scale_fixed_ranges, so they just pass through
                ("fixed_range", "passthrough", list(FIXED_RANGES)),
                ("onehot", OneHotEncoder(drop="first", sparse_output=False, dtype=int), CATEGORICAL_COLUMNS),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        preprocessor.set_output(transform="pandas")
        return preprocessor

    @staticmethod
    def _select_features(df: pd.DataFrame) -> pd.DataFrame:
        """Drops id/gender/target if present, checks that every needed column exists, and fixes column order."""
        df = df.drop(columns=DROP_COLUMNS + [TARGET], errors="ignore")
        missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")
        return df[FEATURE_COLUMNS].copy()

    @staticmethod
    def _unwrap_bedtime(df: pd.DataFrame) -> pd.DataFrame:
        """Shifts bedtimes after midnight past 24 (0.5 -> 24.5) so late bedtimes sit next to 23.5, not far from it."""
        df = df.copy()
        hours = df[BEDTIME_COLUMN]
        df[BEDTIME_COLUMN] = hours.where(hours >= BEDTIME_DAY_START, hours + 24)
        return df

    @classmethod
    def _prepare_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Everything that happens to raw rows before the ColumnTransformer."""
        return cls._scale_fixed_ranges(cls._unwrap_bedtime(cls._select_features(df)))

    @staticmethod
    def _scale_fixed_ranges(df: pd.DataFrame) -> pd.DataFrame:
        """Scales the columns that have known bounds"""
        df = df.copy()
        for col, (low, high) in FIXED_RANGES.items():
            df[col] = (df[col] - low) / (high - low)
        return df

    def _require_fitted(self) -> None:
        try:
            check_is_fitted(self.preprocessor)
        except NotFittedError as e:
            raise RuntimeError(
                "The preprocessor is not fitted yet. Call split_and_save_data() first."
            ) from e

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Turns raw rows into model-ready features using the fitted preprocessor.
        The target column is not included in the output."""
        self._require_fitted()
        return self.preprocessor.transform(self._prepare_features(df))

    def transform_row(self, row: dict | pd.Series | pd.DataFrame) -> pd.DataFrame:
        """Turns ONE raw row into a one-row DataFrame of model-ready features.

        The row needs a value for every feature column, keyed by column name. student_id, gender and cgpa are
        ignored if present, so a full row from the raw dataset can be passed in as is.
        """
        if isinstance(row, dict):
            df = pd.DataFrame([row])
        elif isinstance(row, pd.Series):
            df = row.to_frame().T
        else:
            df = row.copy()
        if len(df) != 1:
            raise ValueError(f"Expected exactly one row, got {len(df)}")

        df = self._select_features(df)

        # validate the input
        df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].apply(pd.to_numeric)  # raises if a number is given as text
        columns_with_nan = df.columns[df.isna().any()].tolist()
        if columns_with_nan:
            raise ValueError(f"Missing values are not allowed, found them in: {columns_with_nan}")
        for col, (low, high) in FIXED_RANGES.items():
            value = df[col].iloc[0]
            if not low <= value <= high:
                raise ValueError(f"{col} must be between {low} and {high}, got {value}")

        return self.transform(df)

    # ----------------------------------------------------------------------------------
    # Splitting, saving, loading
    # ----------------------------------------------------------------------------------
    def split_and_save_data(self, save_folder_path: str, train_frac: float = 0.70,
                val_frac: float = 0.15,
                test_frac: float = 0.15,
                random_state: int = 42) -> Self:
        """Splits the cleaned data, fits the preprocessor on the training split, and saves:
        train.csv, validation.csv, test.csv (scaled/encoded features + raw cgpa) and preprocessor.joblib"""
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
        # fit on the TRAINING set only
        self.preprocessor.fit(self._prepare_features(train_df))

        # transform every split with the same fitted preprocessor, then save
        self.splits = {}
        for name, raw_df in {"train": train_df, "validation": val_df, "test": test_df}.items():
            processed = self.transform(raw_df)
            processed[TARGET] = raw_df[TARGET]  # target stays in its original units (0-4 GPA), unscaled
            processed.to_csv(os.path.join(save_folder_path, f"{name}.csv"), index=False)
            self.splits[name] = processed

        # save the fitted preprocessor so single rows can be transformed later without the raw data
        joblib.dump(self.preprocessor, os.path.join(save_folder_path, PREPROCESSOR_FILENAME))
        print(f"Datasets and preprocessor have been saved to {os.path.abspath(save_folder_path)}")
        return self

    def load_preprocessor(self, folder_path: str = PROCESSED_DATA_PATH, file_name: str = PREPROCESSOR_FILENAME) -> Self:
        """Loads the fitted preprocessor that split_and_save_data() saved in that folder."""
        path = os.path.join(_resolve_path(folder_path), file_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved preprocessor named {file_name} was found at {path}. Run the processing pipeline first.")
        self.preprocessor = joblib.load(path)
        return self

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
    # bedtimes after midnight are stored as 0.5 rather than 24.5, which splits late sleepers across both ends of the
    # scale. we shift them past midnight before removing outliers.
    processor.unwrap_bedtime()
    # we remove rows with extreme outlier values that are over 3 standard deviations away from the mean value
    processor.remove_outliers(3.0)
    processor.split_and_save_data(output_folder)
    if verbose:
        print("######   PROCESSED DATA (training split)   ######")
        processor.inspect_data(processor.splits["train"])
    return processor


if __name__ == '__main__':
    run_processing_pipeline(RAW_DATA_PATH, PROCESSED_DATA_PATH, verbose=True)

