# main.py
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

if __name__ == "__main__":
        obj=DataIngestion()
        train_data, test_data=obj.initiate_data_ingestion()
        data_transform=DataTransformation()
        train_arr,test_arr,_ =data_transform.initiate_data_transformation(train_data,test_data)


