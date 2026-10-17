import os.path
from tkinter.filedialog import SaveFileDialog

import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

PROCESSED_DATASET_PATH = "../data/processed/"
SAVED_MODELS_PATH = "../models/"

class ModelTrainer:
    model: LinearRegression
    training_data: pd.DataFrame
    validation_data: pd.DataFrame
    testing_data: pd.DataFrame

    def __init__(self):
        self.model = LinearRegression()

    def load_datasets(self, folder_path: str):
        if not os.path.exists(folder_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            folder_path = os.path.join(script_dir, folder_path)
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"Path {folder_path} does not exist!")

        training_path = os.path.join(folder_path, "train.csv")
        validation_path = os.path.join(folder_path, "test.csv")
        testing_path = os.path.join(folder_path, "validation.csv")

        for filename in [training_path, validation_path, testing_path]:
            assert os.path.isfile(filename), f"Can't find file {filename}"

        self.training_data = pd.read_csv(training_path)
        self.validation_data = pd.read_csv(validation_path)
        self.testing_data = pd.read_csv(testing_path)

    def train_model(self, model=LinearRegression()):
        self.model = model

        X_train = self.training_data.drop(["cgpa"], axis=1)
        y_train = self.training_data["cgpa"]
        X_test = self.validation_data.drop(["cgpa"], axis=1)
        y_test = self.validation_data["cgpa"]
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        print("MAE :", mean_absolute_error(y_test, y_pred))
        print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
        print("R2  :", r2_score(y_test, y_pred))

        # scatter plot of actual vs predicted
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)

        # draw "perfect prediction" line
        lowest = min(y_test.min(), y_pred.min())
        highest = max(y_test.max(), y_pred.max())
        plt.plot([lowest, highest], [lowest, highest], color="red", linestyle="--", label="Perfect prediction")

        plt.xlabel("Actual CGPA")
        plt.ylabel("Predicted CGPA")
        plt.title(f"Predicted vs Actual {self.model.__class__.__name__}")
        plt.legend()
        plt.tight_layout()
        plt.show()

        return self.model

    def save_model(self, path = SAVED_MODELS_PATH, filename="best_model.pk1"):
        if not os.path.exists(path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(script_dir, path)
        if not os.path.exists(path):
            os.makedirs(path)
        joblib.dump(self.model, os.path.join(path, filename))

if __name__ == '__main__':
    trainer = ModelTrainer()
    trainer.load_datasets(PROCESSED_DATASET_PATH)

    print("\nLinear Regression: ")
    trainer.train_model(LinearRegression())
    trainer.save_model(filename="LinearRegression.pkl")
    print("\nRandom Forest Regressor: (n_estimators=200, max_depth=10)")
    trainer.train_model(RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42))
    print("\nGradient Boosting Regressor: (n_estimators=200, max_depth=10)")
    trainer.train_model(GradientBoostingRegressor(n_estimators=200, max_depth=10, random_state=42))