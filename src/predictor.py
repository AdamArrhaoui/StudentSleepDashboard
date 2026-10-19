import os

import joblib
import pandas as pd

from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer

TRAINED_MODEL_PATH = "../models/best_model.pkl"
PREPROCESSOR_DATA_PATH = "../data/processed"
PREPROCESSOR_FILENAME = "preprocessor.joblib"

def _resolve_path(path: str) -> str:
    """Use the path as given if it exists, otherwise treat it as relative to this script's folder."""
    if os.path.exists(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, path)

class Predictor:
    data_processor: DataProcessor
    model = None

    def __init__(self, model_path: str = TRAINED_MODEL_PATH, preprocessor_folder_path:str = PREPROCESSOR_DATA_PATH, preprocessor_filename: str = PREPROCESSOR_FILENAME):
        self.data_processor = DataProcessor()
        self.data_processor.load_preprocessor(preprocessor_folder_path, preprocessor_filename)
        model_path = _resolve_path(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found at {model_path}")
        self.model = joblib.load(model_path)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        transformed_features = self.data_processor.transform_row(features)
        return self.model.predict(transformed_features)
