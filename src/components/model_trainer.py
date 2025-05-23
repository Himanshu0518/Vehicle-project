import sys
import pandas as pd
from pandas import DataFrame

from src.constants import MODEL_TRAINER_MODEL_CONFIG_FILE_PATH , TARGET_COLUMN
from src.entity.config_entity import ModelTrainerConfig 
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact , ModelTrainerArtifact 
from src.exception import MyException
from src.logger import logging

from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class ModelTrainning:
    """ training model using required algorithm """
    def __init__(self):
        
      try:
        self.model_config = ModelTrainerConfig()
        self.model_artifact  = ModelTrainerArtifact()
        self.model_info = read_yaml_file(file_path = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH )
        self.data_transformation_artifact = DataTransformationArtifact()

      except Exception as e:
         raise MyException(e,sys)
      

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)
        

    def train_model(self , train_df:DataFrame ) :
        """ take transformed data and train the model """
        try:
          X_train = train_df.drop(TARGET_COLUMN , axis=1)
          # X_test = test_df.drop(TARGET_COLUMN , axis=1)
          # Y_train = train_df[TARGET_COLUMN]
          # Y_test = test_df[TARGET_COLUMN]
        
        except Exception as e:
         raise MyException(e,sys)
        
        model = self.model_info['model_name']
        try:    
          model.fit(X_train)
        except Exception as e:
           raise MyException(e,sys)
        
        return model 
    
    def initiate_model_trainner(self)->ModelTrainerArtifact :
       
        train_df = self.read_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
        test_df = self.read_data(file_path=self.data_transformation_artifact.transformed_test_file_path) 

        
       

       
    
      




    
 
      

    
         
