import pandas as pd
import logging
import os
from sklearn.model_selection import train_test_split 
logs_dir="logs"
os.makedirs(logs_dir,exist_ok=True)

logger=logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

console_handler=logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(logs_dir, "data_ingestion.log"))
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(path:str)->pd.DataFrame:
    try:
        logger.debug("Data ingestion statred...")
        df=pd.read_csv(path)
        logger.debug("Data loaded successfully from %s",path)
        return df
        
    except FileNotFoundError as e:
        logger.error("No file found")
        raise
    except Exception as e:
        logger.error("Unexpected error occured %s",e)
        raise

def pre_process(df:pd.DataFrame,path:str)->pd.DataFrame:
    try:
        logger.debug("Pre_Processing started...")
        logger.info("Before Pre_Processing shape of the dataframe : %s",df.shape)
        #Assigning headers 
        df.columns = ["ID", "Entity", "Sentiment","Text"]

        # Save back to CSV
        df.to_csv(path, index=False)
        df.drop('ID',axis=1,inplace=True)
        df['Combined_Text']=df['Entity']+" " + df['Text']
        df.drop('Entity',axis=1,inplace=True)
        df.drop('Text',axis=1,inplace=True)
        df.dropna(subset=['Combined_Text'],inplace=True)
        df.drop_duplicates(inplace=True)

        logger.debug("Data Pre_Processing Done...")
        logger.info("After Pre_Processing shape of the dataframe is : %s",df.shape)
        return df

    except KeyError as e:
        logger.error(f"Missing column error during preprocessing: {e}")
        raise
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise

def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)->None:
    try:
        logger.debug("Saving Data...")
        raw_data_path=os.path.join(data_path,"raw_data")
        os.makedirs(raw_data_path,exist_ok=True)

        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug(f"Data saved successfully to {data_path}")
    except Exception as e:
        logger.error(f"Error saving data to {data_path}: {e}")
        raise



def main():
    path ="https://raw.githubusercontent.com/Tanuj-code-b2006/Datasets/refs/heads/main/twitter_training.csv"
    df=load_data(path)
    final_df=pre_process(df,path)
    test_size=0.4
    train_data,test_data=train_test_split(final_df,test_size=test_size,random_state=2)
    save_data(train_data, test_data, data_path='./data')
if __name__=="__main__" :
    main()