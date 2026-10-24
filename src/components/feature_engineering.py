import pandas as pd
import os
import logging
import yaml
from sklearn.feature_extraction.text import CountVectorizer

logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)

logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.DEBUG)  

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(logs_dir, "feature_engineering.log"))
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_path:str)->pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        logger.info("Data loaded successfully.")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Parser error while loading data from {data_path}: {e}")
        raise


def apply_count_vectorizer(train_data:pd.DataFrame,
                           test_data:pd.DataFrame,
                           ngram_range:tuple,
                           min_df:int,
                           max_features:int)->pd.DataFrame:
    try:
        logger.debug("Applying Bag of Words for vectorization...")
        vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features
        )
        X_train=train_data['Combined_Text'].fillna("").astype(str)
        X_test=test_data['Combined_Text'].fillna("").astype(str)
        y_train=train_data['Sentiment']
        y_test=test_data['Sentiment']
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        train_df=pd.DataFrame(X_train_bow.toarray())
        train_df['Sentiment']=y_train
        test_df=pd.DataFrame(X_test_bow.toarray())
        test_df['Sentiment']=y_test
        logger.debug("Bag of words vectorization completed.")

        return train_df,test_df

    except Exception as e:
        logger.error("Unexpected error occured during vectorization %s",e)
        raise

def save_data(df: pd.DataFrame, path: str) -> None:
        """
        Function to save the DataFrame to a CSV file.

        Args:
            df (pd.DataFrame): The DataFrame to be saved.
            path (str): The path where the DataFrame will be saved.
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)
            logger.debug(f"Data saved successfully to {path}.")
        except Exception as e:
            logger.error(f"Error saving data to {path}: {e}")
            raise


def main():
    ngram_range=(1,2)
    min_df=2
    max_features=18000
    try:
        train_data=load_data("data/interim_data/train_preprocessed.csv")
        test_data=load_data("data/interim_data/test_preprocessed.csv")

        train_df,test_df=apply_count_vectorizer(train_data,test_data,ngram_range,min_df,max_features)

        save_data(train_df, os.path.join("./data", "processed_data", "train_bow.csv"))
        save_data(test_df, os.path.join("./data", "processed_data", "test_bow.csv"))
    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()