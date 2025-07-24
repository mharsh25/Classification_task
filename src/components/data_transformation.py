import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler, LabelEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_tranformation_config=DataTransformationConfig()
    
    def get_data_tranformer_object(self):
        ''''
        This function is reponsible for data transformation
        '''
        try:
            numerical_columns=['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
            categorical_columns=[
                'Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE',
       'SCC', 'CALC', 'MTRANS'

            ]
            
            target_col=['NObeyesdad']

            num_pipepine=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
                )
            
            
            logging.info("Numerical columns scaling completed")
            logging.info("Categorical columns encoding completed")
            logging.info("Target column Encoding completed")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipepine,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns),
                    
                ]
            )

            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self, train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("read train and test data completed")

            logging.info("obtaining Preprocessing object")

            preprocessing_obj=self.get_data_tranformer_object()
            
            target_column_name="NObeyesdad"
            numerical_columns=['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
            train_df_features=train_df.loc[:,train_df.columns!=target_column_name]
            test_df_features=test_df.loc[:,test_df.columns!=target_column_name]

            logging.info(
                    f"Applying preprocessing object on training dataframe and testing dataframe"
            )

            le=LabelEncoder()
            train_arr_features=preprocessing_obj.fit_transform(train_df_features)
            train_arr_target=le.fit_transform(train_df[target_column_name])
            train_arr=np.c_[
                train_arr_features,
                train_arr_target]

            test_arr_featues=preprocessing_obj.fit_transform(test_df_features)
            test_arr_target=le.fit_transform(test_df[target_column_name])
            test_arr=np.c_[
                test_arr_featues,
                test_arr_target]
            
            logging.info(f"Saved preprocessing object.")
            
            save_object(
                file_path=self.data_tranformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)