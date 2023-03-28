'''
This file contains common functionality such as saving object pickle files,etc
This file and functions are called in data transformation and training scripts primarily.
'''


import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException

def save_object(file_path, object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(object, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            file = dill.load(file_obj)
        return file
        
    except Exception as e:
        raise CustomException(e, sys)