import os.path

from sklearn.linear_model import LinearRegression
import pandas as pd

PROCESSED_DATASET_PATH = "../data/processed/"

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
        self.validation_data = pd.read_csv(training_path)
        self.testing_data = pd.read_csv(training_path)

if __name__ == '__main__':
    trainer = ModelTrainer()
    trainer.load_datasets(PROCESSED_DATASET_PATH)
    print(trainer.testing_data.head())