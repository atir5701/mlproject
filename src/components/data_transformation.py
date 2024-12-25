import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline




@dataclass
class DataTransformationConfig:
    preprocessor_density_obj_path = os.path.join("assets","preprocessorDensity.pkl")
    preprocessor_viscosity_obj_path = os.path.join("assets","preprocessorViscosity.pkl")


class DataTransformation:
    def __init__(self):
        self.data_trans_config = DataTransformationConfig()

    @staticmethod
    def outlierRemoval(X):
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return np.clip(X, lower_bound, upper_bound)

    def get_data_transformer_object_density(self):
        try:
            num_feature = ['MR', 'T/K', 'MCI', 'ωm', 'MW']
            numeric_transformer = StandardScaler()
            numeric_inputer = SimpleImputer(strategy="median")
            clamping_transformer = FunctionTransformer(self.outlierRemoval, validate=False)
            preprocessor = ColumnTransformer(
                [
                ("SimpleImputer",numeric_transformer,num_feature),   
                ("StandardScaler",numeric_transformer,num_feature),
                ("outlierRemoval",clamping_transformer,num_feature)
                ]
            )
            pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
            logging.info("Data Preprocessing Pipeline for Density created.")

            return pipeline

        except Exception as e:
            raise CustomException(e,sys)    

    def get_data_transformer_object_viscosity(self):
        try:
            num_feature = ['inv_Temp','MCI','Avg Molar',' omega']
            numeric_transformer = StandardScaler()
            numeric_inputer = SimpleImputer(strategy="median")
            clamping_transformer = FunctionTransformer(self.outlierRemoval, validate=False)
            preprocessor = ColumnTransformer(
                [
                ("SimpleImputer",numeric_transformer,num_feature),   
                ("StandardScaler",numeric_transformer,num_feature),
                ("outlierRemoval",clamping_transformer,num_feature),
                ]
            )
            pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
            logging.info("Data Preprocessing Pipeline for Viscosity created.")
            return pipeline

        except Exception as e:
            raise CustomException(e,sys)    

    def initiale_data_transformation_density(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and Test Data loaded")
            
            pipeline = self.get_data_transformer_object_density()
            
            logging.info("Preprocessing pipeline obtained for Density")
            
            target_column_name = "Dexp"
            
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying Preprocessing on Density data.")

            input_feature_train = pipeline.fit_transform(input_feature_train_df)
            input_feature_test = pipeline.transform(input_feature_test_df)

            train_arr =  np.c_[
                input_feature_train,np.array(target_feature_train_df)
            ]

            test_arr =  np.c_[
                input_feature_test,np.array(target_feature_test_df)
            ]

            logging.info("Save the preprocessing object of Density.")

            save_object(
                file_path = self.data_trans_config.preprocessor_density_obj_path,
                obj = pipeline
            )

            return (train_arr,test_arr,self.data_trans_config.preprocessor_density_obj_path)
  
        except Exception as e:
            raise CustomException(e,sys)    
        
    def initiale_data_transformation_viscosity(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and Test Data loaded")
            
            pipeline = self.get_data_transformer_object_viscosity()
            
            logging.info("Preprocessing pipeline obtained for Viscosity")
            
            target_column_name = "V/cP"
            
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying Preprocessing on Viscosity data.")

            input_feature_train = pipeline.fit_transform(input_feature_train_df)
            input_feature_test = pipeline.transform(input_feature_test_df)

            train_arr =  np.c_[
                input_feature_train,np.array(target_feature_train_df)
            ]

            test_arr =  np.c_[
                input_feature_test,np.array(target_feature_test_df)
            ]

            logging.info("Save the preprocessing object of Viscosity.")

            save_object(
                file_path = self.data_trans_config.preprocessor_viscosity_obj_path,
                obj = pipeline
            )

            return (train_arr,test_arr,self.data_trans_config.preprocessor_viscosity_obj_path)
  
        except Exception as e:
            raise CustomException(e,sys)         