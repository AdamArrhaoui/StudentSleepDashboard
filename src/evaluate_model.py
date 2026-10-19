# Import time
import time
# Import path from pathlib
from pathlib import Path

import joblib
# Import pandas 
import pandas as pd
# Import models from SKLEARN
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Import metrics from sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import model trainer
from src.model_trainer import ModelTrainer

# Establish project root using pathlib
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Build the direcotry for the processed data
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
# Build the path for the models directory
MODELS_DIR = PROJECT_ROOT / "models"
# Define the target
TARGET = "cgpa"

# Same models and settings as model_trainer.py, so the table matches what was trained
CANDIDATES = [
    ("Linear Regression", "Default", LinearRegression()),
    ("Random Forest", "200 trees, max depth 10",
     RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)),
    ("Random Forest (tuned)", "300 trees, max depth 5, min leaf 10, sqrt features",
     RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=10,
                           max_features="sqrt", random_state=42)),
    ("Gradient Boosting", "200 trees, max depth 10",
     GradientBoostingRegressor(n_estimators=200, max_depth=10, random_state=42)),
    ("Gradient Boosting (tuned)", "200 trees, max depth 2, min leaf 10, lr 0.05",
     GradientBoostingRegressor(n_estimators=200, max_depth=2, min_samples_leaf=10,
                               learning_rate=0.05, random_state=42)),
]

# The model saved in models/best_model.pkl (chosen by its validation score); the dashboard uses this one
SELECTED_MODEL = "Random Forest (tuned)"

# Main function 
def main():
    """Train every candidate, score each one once on the test set, and save the results for the dashboard."""
    trainer = ModelTrainer()
    trainer.load_datasets(str(PROCESSED_DIR))

    # testing_data is validation.csv: the paths in load_datasets are swapped, and the model was chosen
    # using test.csv, so validation.csv is the only split never used to pick a model
    X_test = trainer.testing_data.drop(columns=TARGET)
    y_test = trainer.testing_data[TARGET]

    # Step 2: train, time and score each model (one dictionary per model)
    rows = []
    for name, config, model in CANDIDATES:
        start = time.perf_counter()
        trainer.train_model(model)  # partner's method: fits the model on the training set
        train_time = time.perf_counter() - start

        y_pred = model.predict(X_test)
        rows.append({
            "model": name,
            "configuration": config,
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": mean_squared_error(y_test, y_pred) ** 0.5,
            "r2": r2_score(y_test, y_pred),
            "train_time_s": train_time,
        })

    # Step 3: plot data for the saved model (Predicted vs. actual, Residuals and Features tabs)
    saved_model = joblib.load(MODELS_DIR / "best_model.pkl")
    y_pred = saved_model.predict(X_test)

    pd.DataFrame({"actual": y_test, "predicted": y_pred}).to_csv(
        MODELS_DIR / "test_predictions.csv", index=False)

    # Tree-based models learn one importance value per feature
    pd.DataFrame({"feature": X_test.columns, "importance": saved_model.feature_importances_}).to_csv(
        MODELS_DIR / "feature_importance.csv", index=False)

    # Step 4: the comparison table, with the saved model marked as "selected"
    metrics = pd.DataFrame(rows)
    metrics["selected"] = metrics["model"] == SELECTED_MODEL
    metrics.to_csv(MODELS_DIR / "metrics.csv", index=False)

    print("\nTest-set results:")
    print(metrics.round(3).to_string(index=False))
    print(f"\nSaved metrics.csv, test_predictions.csv and feature_importance.csv to {MODELS_DIR}")


# Only run when executed directly (python -m src.evaluate_model), not when imported
if __name__ == "__main__":
    main()


