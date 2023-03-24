import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@dataclass
class DataTransformationConfig:
    preprocessor_object_filepath: str= os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformer_object(self):
        try:
            num_cols = ['reading_score', 'writing_score']
            cat_cols = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                           ('scaler', StandardScaler())])
            cat_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                           ('onehot', OneHotEncoder())])

            preprocessor = ColumnTransformer([('numerical', num_pipeline, num_cols),
                                             ('categorical', cat_pipeline, cat_cols)])
            logging.info('Preprocessor object built successfully.')
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # initialize the transformer object and apply
            preprocessor_obj = self.get_data_transformer_object()

            num_cols = ['reading_score', 'writing_score']
            cat_cols = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            target_col = ['math_score']

            X_train_df = train_df.drop(target_col, axis=1)
            Y_train_df = train_df[target_col]

            X_test_df = test_df.drop(target_col, axis=1)
            Y_test_df = test_df[target_col]

            X_train_preprocess = pd.DataFrame(preprocessor_obj.fit_transform(X_train_df))
            train_preprocess = pd.concat([X_train_preprocess, Y_train_df], axis=1)
            logging.info('Preprocessed Train Data')

            X_test_preprocess = pd.DataFrame(preprocessor_obj.transform(X_test_df))
            test_preprocess = pd.concat([X_test_preprocess, Y_test_df], axis=1)
            logging.info('Preprocessed Test Data')

            # save the tranformer object to pkl file 
            save_object(self.data_transformation_config.preprocessor_object_filepath, preprocessor_obj)
            logging.info('Saved Preprocessor Object file to Disk!')

            # return transformed train, test and transformer pickle file path
            return (train_preprocess,
                    test_preprocess,
                    self.data_transformation_config.preprocessor_object_filepath)
        
        except Exception as e:
            raise CustomException(e, sys)