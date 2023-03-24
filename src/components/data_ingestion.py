import os
import sys
from src.exception import CustomException
from src.logger import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation

@dataclass #Automatically initializes the special methods like __init__() under that class
class DataIngestionConfig():
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion():
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_dataingestion(self):
        logging.info('Entered Data Ingestion Method')
        try:
            data = pd.read_csv('notebooks\data\stud.csv')
            logging.info('Reading dataset complete!')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            data.to_csv(self.ingestion_config.raw_data_path, header=True, index=False)
            logging.info('Saved dataset to Disk!')

            logging.info('Train Test Split Start')
            train, test = train_test_split(data, test_size=0.2, random_state=1)

            train.to_csv(self.ingestion_config.train_data_path,  header=True, index=False)
            test.to_csv(self.ingestion_config.test_data_path,  header=True, index=False)
            logging.info('Saved Train and Test data to Disk!')
            logging.info('Data Ingestion Complete!')

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

        except:
            pass

if __name__ == '__main__':
    obj = DataIngestion()
    train_path, test_path = obj.initiate_dataingestion()

    transformation_obj = DataTransformation()
    train_df, test_df, preprocess_obj_path = transformation_obj.initiate_data_transformation(train_path, test_path)