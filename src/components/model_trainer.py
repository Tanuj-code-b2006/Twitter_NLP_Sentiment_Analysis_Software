import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.svm import LinearSVC



logs_dir='logs'
os.makedirs(logs_dir,exist_ok=True)
logger=logging.getLogger("model_trainer")
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(logs_dir, "model_building.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)








def load_data(path:str)->pd.DataFrame:
    try:
        logger.debug("Data loading from %s",path)
        df=pd.read_csv(path)
        logger.debug("Data loaded from %s with shape %s", path, df.shape)
        return df

    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s", e)
        raise

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        raise

    except Exception as e:
        logger.error(
            "Unexpected error occurred while loading the data: %s", e
        )
        raise

def train_model(X_train: np.ndarray,
                y_train: np.ndarray,
                penalty:str,
                loss:str,
                max_iter:int,
                random_state:int)->LinearSVC:
    try:
        if X_train.shape[0]!=y_train.shape[0]:
            raise ValueError("number of samples must be same")

        logger.debug("Model Training started...")
        clf=clf = LinearSVC(
            penalty=penalty,
            loss=loss,
            max_iter=max_iter,
            random_state=random_state,

)
        clf.fit(X_train,y_train)

        logger.debug("Model Training done...")
        return clf
    except FileNotFoundError as e:
        logger.error("File not found error occured %s",e)

    except Exception as e:
        logger.error("Unexpected error occured %s",e)

def save_model(model, file_path: str) -> None:
    
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file:
            pickle.dump(model, file)

        logger.debug("Model saved to %s", file_path)

    except FileNotFoundError as e:
        logger.error("File path not found: %s", e)
        raise

    except Exception as e:
        logger.error(
            "Error occurred while saving ")
        raise


def main():
    try:

        train_data = load_data(
            "./data/processed_data/train_bow.csv"
        )

        x_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(
            x_train,
            y_train,
            penalty='l2',
            loss='hinge',
            max_iter=5000,
            random_state=42
        )

        model_save_path = "models/model.pkl"
        save_model(clf, model_save_path)

    except Exception as e:
        logger.error(
            "Failed to complete the model building process: %s",e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()