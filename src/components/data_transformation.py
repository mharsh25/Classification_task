import sys
import os
from dataclasses import dataclass
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

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
            
            #target_col=['NObeyesdad']

            num_pipepine=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore')),
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
            df=pd.concat([train_df,test_df],axis=0)
            logging.info("read train and test data completed")

            logging.info("obtaining Preprocessing object")

            preprocessing_obj=self.get_data_tranformer_object()
            
            target_column_name="NObeyesdad"
            #numerical_columns=['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
            df_features=df.loc[:,df.columns!=target_column_name]
            #print(train_df_features.shape)
            #print(test_df_features.shape)
            logging.info(
                    f"Applying preprocessing object on training dataframe and testing dataframe"
            )

            le=LabelEncoder()
            arr_features=preprocessing_obj.fit_transform(df_features)
            arr_target=le.fit_transform(df[target_column_name])
            
            train_arr_features,test_arr_features,train_target_arr,test_target_arr=train_test_split(arr_features,arr_target, test_size=.2,random_state=42)

            train_arr=np.c_[
                train_arr_features,
                train_target_arr
            ]
            test_arr=np.c_[
                test_arr_features,
                test_target_arr
            ]
            
            logging.info(f"Saved preprocessing object.")
            
            save_object(
                file_path=self.data_tranformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            with open('artifacts/label_encoder.pkl', 'wb') as f:
                    pickle.dump(le, f)

            logging.info("Saving Label Encoder file")


            return(
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_obj_file_path
                
            )
        except Exception as e:
            raise CustomException(e,sys)