import os
import sys
import dill 

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)        
#def evaluate_model(true, predicted ):
 #   precision=precision_score(true, predicted, average='macro')
  #  recall=recall_score(true, predicted, average='macro')
   # accuracy=accuracy_score(true, predicted)

    #return precision, recall, accuracy

    

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)