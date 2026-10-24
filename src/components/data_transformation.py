import pandas as pd
import logging
import os
import nltk
import re
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelEncoder


logs_dir="logs"
os.makedirs(logs_dir,exist_ok=True)

logger=logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

console_handler=logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(logs_dir, "data_transformation.log"))
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def transform_text(text: str) -> str:
    """
    Clean and preprocess text:
    1. Remove HTML tags
    2. Remove URLs and emails
    3. Convert to lowercase
    4. Remove special characters
    5. Tokenize
    6. Remove stopwords
    7. Apply stemming
    """

    try:
        logging.info("Transforming text...")

        if text is None:
            return ""

        # Convert to string
        text = str(text)

        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()

        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

        # Remove emails
        text = re.sub(r'\S+@\S+', ' ', text)

        # Lowercase
        text = text.lower()

        # Keep only alphabets and spaces
        text = re.sub(r'[^a-z\s]', ' ', text)

        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenization
        tokens = nltk.word_tokenize(text)

        # Stopword removal
        stop_words = set(nltk.corpus.stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

        # Stemming
        stemmer = nltk.stem.PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]

        return " ".join(tokens)

    except Exception as e:
        logging.error(f"Error in transform_text: {e}")
        raise e

def preprocess_data(df, text_column='Combined_Text', target_column='Sentiment', label_encoder=None):

    try:
        logger.debug("Starting data preprocessing.")

        if label_encoder is None:
            label_encoder = LabelEncoder()
            df[target_column] = label_encoder.fit_transform(df[target_column])
        else:
            df[target_column] = label_encoder.transform(df[target_column])
        logger.debug("Encoded target column...")

        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug("Transformed text data.")
        return df
    except KeyError as e:
        logger.error(f"Missing column error during preprocessing: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

def main(text_column='Combined_Text', target_column='Sentiment'):
    """
    Main function to execute the preprocessing steps.

    Args:               
        text_column (str): The name of the column containing the text data.
        target_column (str): The name of the column containing the target data.
    """
    try:
        logger.debug("Starting main preprocessing function.")

        train_data = pd.read_csv("data/raw_data/train.csv")
        test_data = pd.read_csv("data/raw_data/test.csv")
        logger.debug("Loaded training and testing data.")

        le = LabelEncoder()
        le.fit(train_data[target_column])

        train_data = preprocess_data(train_data, text_column, target_column, le)
        test_data = preprocess_data(test_data, text_column, target_column, le)

        data_path = os.path.join("./data", "interim_data")
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train_preprocessed.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test_preprocessed.csv"), index=False)
        logger.debug("Saved preprocessed data to data/interim_data directory.")

    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        raise
    except KeyError as e:
        logger.error(f"Missing column error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in main preprocessing function: {e}")
        raise


if __name__ == "__main__":
    main()