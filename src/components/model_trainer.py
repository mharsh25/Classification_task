import os
import sys
from dataclasses import dataclass

#from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestClassifier
)

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
#from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
#from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                    "Logistic Regression":LogisticRegressionCV(max_iter=1000),
                    "SVM":SVC(kernel='linear'),
                    "Random Forest":RandomForestClassifier(criterion='log_loss',n_jobs=-1)
                    }

            params={
                "Logistic Regression": {
                    'Cs':[5,10,15,20,25,50]
                },
                "Random Forest":{
                    
                    'n_estimators': [8,16,32,64],
                    'max_features': ['sqrt', 'log2']
                },
                "SVM":{
                    "C":[0.5, 1.0, 10],
                    "kernel":['linear', 'poly']
                }
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)

            ##to get best model score from dict
            best_model_score=max(sorted(model_report.values()))

            #To get best model name from dict

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found on both train and test dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            accuracy_scr=accuracy_score(y_test,predicted)
            return accuracy_scr

        except Exception as e:
            raise CustomException(e,sys)