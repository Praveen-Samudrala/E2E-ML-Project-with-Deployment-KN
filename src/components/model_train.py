import os
import sys
from src.exception import CustomException
from src.logger import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMRegressor

from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str=os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.target_col = ['math_score']

    def evaluate_models(self, x_train, y_train, x_test, y_test, models):
        try:
            scores_report = {}
            for name, model in models.items():
                # train
                model.fit(x_train, y_train.values.ravel())
                # Getting a warning DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
                # Hence modified y_train to y_train.values.ravel(). Both of them had shape of (1000,) but don't know what's the warning specificlaly about.
                
                # predict
                y_test_pred = model.predict(x_test)

                # Scores
                test_r2 = r2_score(y_test, y_test_pred)
                
                # Store in dictionary
                scores_report[name] = test_r2

            return scores_report
        
        except Exception as e:
            raise CustomException(e, sys)
            

    def initiate_model_training(self, x_train_df, y_train_df, x_test_df, y_test_df):
        try:
            logging.info('Training Process Started!')

            x_train, y_train, x_test, y_test = (x_train_df, y_train_df, x_test_df, y_test_df)
            logging.info('Featched Preprocessed Train Test Data.')

            # create models dict
            models = {'Linear Regression' : LinearRegression(),
                        'Lasso Regression' : Lasso(),
                        'Ridge Regression' : Ridge(),
                        'Decision Trees': DecisionTreeRegressor(),
                        'Random Forest' : RandomForestRegressor(),
                        'Ada Boosting' : AdaBoostRegressor(),
                        'Light GBM' : LGBMRegressor()}
            
            # Create dict of model evaluation scores - write evaluation function in utils
            model_score_report: dict = self.evaluate_models(x_train, y_train, x_test, y_test, models)
            print(model_score_report)
            logging.info('Model Training and Evaluation complete.')

            best_model_name = sorted(model_score_report, key= lambda x: model_score_report[x], reverse=True)[0]
            best_model_score = model_score_report[best_model_name]
            print(f"Best Model chosen is: {best_model_name} for Test R2 score: {best_model_score}")

            if best_model_score < 0.6:
                raise CustomException('No best model found.')
            logging.info('Found Best model trained on train and evaluated on test data.')

            best_model = models[best_model_name]
            
            save_object(file_path = self.model_trainer_config.trained_model_file_path, 
                        object=best_model)
            
            y_pred = best_model.predict(x_test)
            pred_r2_score = r2_score(y_pred, y_test)
            return pred_r2_score
        
        except Exception as e:
            raise CustomException(e, sys)
