"""Loads the trained model and its results for the Performance and Predict pages.

The dashboard never trains anything itself. The training code (src/model_trainer.py)
saves its results into models/, and these functions read them back. Expected files:

    models/metrics.csv           one row per model:
                                 model, configuration, mae, rmse, r2, train_time_s
    models/test_predictions.csv  best model on the test set: actual, predicted
    models/feature_importance.csv  (optional) feature, importance
    models/loss_history.csv      (optional, neural network) epoch, loss, val_loss

All CGPA values in these files are on the real 0-4 scale (not the scaled 0-1 one).

For live predictions, src/predictor.py must provide a Predictor class:
    Predictor().predict(features: pd.DataFrame) -> array of CGPA values (0-4)
where `features` has the same columns as the cleaned data (real units, text
categories), and the Predictor applies the same scaling and encoding as training.

Drafted with help from Claude (Anthropic).
"""
# Import sys and Path from pathlib to manipulate the system path for module imports.
from pathlib import Path

# Import pandas as pd and streamlit as st for data manipulation and web app functionality.
import pandas as pd
import streamlit as st

# Define the project root directory and the models directory where trained model results are stored. The project root is determined by resolving the parent directory of the current file, and the models directory is a subdirectory named "models" within the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define the models directory where trained model results are stored. The models directory is a subdirectory named "models" within the project root.
MODELS_DIR = PROJECT_ROOT / "models"

# Define the paths to the expected model result files within the models directory. These files include metrics.csv, test_predictions.csv, feature_importance.csv, and loss_history.csv, which contain information about the trained models, their performance on the test set, feature importance, and loss history for neural networks.
METRICS_PATH = MODELS_DIR / "metrics.csv"
# Define the paths to the expected model result files within the models directory. These files include metrics.csv, test_predictions.csv, feature_importance.csv, and loss_history.csv, which contain information about the trained models, their performance on the test set, feature importance, and loss history for neural networks.
PREDICTIONS_PATH = MODELS_DIR / "test_predictions.csv"
# Define the paths to the expected model result files within the models directory. These files include metrics.csv, test_predictions.csv, feature_importance.csv, and loss_history.csv, which contain information about the trained models, their performance on the test set, feature importance, and loss history for neural networks.
IMPORTANCE_PATH = MODELS_DIR / "feature_importance.csv"
# Define the paths to the expected model result files within the models directory. These files include metrics.csv, test_predictions.csv, feature_importance.csv, and loss_history.csv, which contain information about the trained models, their performance on the test set, feature importance, and loss history for neural networks.
LOSS_HISTORY_PATH = MODELS_DIR / "loss_history.csv"


# Write a magic function to read a CSV file if it exists, returning a pandas DataFrame or None if the file does not exist. This function is used to load the model result files (metrics, test predictions, feature importance, and loss history) in a safe manner, avoiding errors when the files are missing.
def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    # Check if the specified path exists. If it does, read the CSV file into a pandas DataFrame and return it. If the path does not exist, return None to indicate that the file is missing.
    return pd.read_csv(path) if path.exists() else None

# Write a load_metrics function to load the metrics.csv file, returning a pandas DataFrame or None if the file does not exist. This function is decorated with @st.cache_data to cache the result, improving performance by avoiding repeated file reads when the data has not changed.
# It uses st.cache_data to cache the result, improving performance by avoiding repeated file reads when the data has not changed.
@st.cache_data
def load_metrics() -> pd.DataFrame | None:
    # Call the _read_csv_if_exists function to read the metrics.csv file from the models directory. If the file exists, it will return a pandas DataFrame containing the model metrics; otherwise, it will return None.
    return _read_csv_if_exists(METRICS_PATH)

# Write a load_test_predictions function to load the test_predictions.csv file, returning a pandas DataFrame or None if the file does not exist. This function is decorated with @st.cache_data to cache the result, improving performance by avoiding repeated file reads when the data has not changed.
# It uses st.cache_data to cache the result, improving performance by avoiding repeated file reads when
@st.cache_data
def load_test_predictions() -> pd.DataFrame | None:
    # Call the _read_csv_if_exists function to read the test_predictions.csv file from the models directory. If the file exists, it will return a pandas DataFrame containing the test predictions; otherwise, it will return None.
    return _read_csv_if_exists(PREDICTIONS_PATH)

# Write a load_feature_importance function to load the feature_importance.csv file, returning a pandas DataFrame or None if the file does not exist. This function is decorated with @st.cache_data to cache the result, improving performance by avoiding repeated file reads when the data has not changed.
@st.cache_data
def load_feature_importance() -> pd.DataFrame | None:
    # Call the _read_csv_if_exists function to read the feature_importance.csv file from the models directory. If the file exists, it will return a pandas DataFrame containing the feature importance values; otherwise, it will return None.
    return _read_csv_if_exists(IMPORTANCE_PATH)

# Write a load_loss_history function to load the loss_history.csv file, returning a pandas DataFrame or None if the file does not exist. This function is decorated with @st.cache_data to cache the result, improving performance by avoiding repeated file reads when the data has not changed.
@st.cache_data
def load_loss_history() -> pd.DataFrame | None:
    # Call the _read_csv_if_exists function to read the loss_history.csv file from the models directory. If the file exists, it will return a pandas DataFrame containing the loss history values; otherwise, it will return None.
    return _read_csv_if_exists(LOSS_HISTORY_PATH)


@st.cache_resource  # "resource": one shared model object, not copied like data
def load_predictor():
    """Return (predictor, None) when the model is ready, or (None, reason) when it isn't."""
    # Try to import the Predictor class from src/predictor.py. If the import fails, return None and a message indicating that the Predictor class is not defined yet. If the import succeeds, attempt to create an instance of the Predictor class. If a FileNotFoundError occurs (indicating that the model files haven't been saved yet), return None and a message explaining the issue. Otherwise, return the Predictor instance and None to indicate that the model is ready for use.
    try:
        # Import the Predictor class from src/predictor.py. This class is expected to provide a predict method that takes a DataFrame of features and returns an array of predicted CGPA values (0-4 scale).
        from src.predictor import Predictor
        # If the import fails, return None and a message indicating that the Predictor class is not defined yet.
    except ImportError:
        return None, "src/predictor.py doesn't define a `Predictor` class yet."
    # If the import succeeds, attempt to create an instance of the Predictor class. If a FileNotFoundError occurs (indicating that the model files haven't been saved yet), return None and a message explaining the issue. Otherwise, return the Predictor instance and None to indicate that the model is ready for use.
    try:
        # Create an instance of the Predictor class. This instance will be used to make predictions on new data using the trained model.
        return Predictor(), None
    # If a FileNotFoundError occurs (indicating that the model files haven't been saved yet), return None and a message explaining the issue. This allows the application to handle the situation gracefully without crashing.
    except FileNotFoundError as error:
        # Return None and a message indicating that the model files haven't been saved yet, along with the specific error message. This helps users understand why the predictor is not available and what they can do to resolve the issue (e.g., running the training code to generate the model files).
        return None, f"The model files haven't been saved yet ({error})."
