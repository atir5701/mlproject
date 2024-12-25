import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import os
import sys

@dataclass
class ModelTrainerConfig:
    trained_density_model_file_path = os.path.join("assets","densityModel.pkl")
    trained_viscosity_model_file_path = os.path.join("assets","viscosityModel.pkl")


class ModelTrainer:
    
    def __init__(self):
        self.trainer_config = ModelTrainerConfig()

    @staticmethod
    def evaluate_models(X_train, y_train,X_test,y_test,models):
        try:
            report = {}

            for i in range(len(list(models))):
                model = list(models.values())[i]
                
                model.fit(X_train,y_train)

                y_test_pred = model.predict(X_test)


                test_model_score = r2_score(y_test, y_test_pred)

                report[list(models.keys())[i]] = test_model_score

            return report

        except Exception as e:
            raise CustomException(e, sys)

    def initiateDensityModelTrainer(self,train_data,test_data):
        try:
            logging.info("Entered the training for density Model.")
            X_train,y_train = (train_data[:,:-1],train_data[:,-1])
            X_test,y_test = (test_data[:,:-1],test_data[:,-1])

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            report = self.evaluate_models(X_train,y_train,X_test,y_test,models)

            best_score = max(list(report.values()))

            best_model_name = list(report.keys())[list(report.values()).index(best_score)]

            best_model_obj = models[best_model_name]

            logging.info("Best Model obtained for density")


            save_object(file_path=self.trainer_config.trained_density_model_file_path,obj=best_model_obj)

            return (best_model_name,best_score)
        

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiateViscosityModelTrainer(self,train_data,test_data):
        try:
            logging.info("Entered the training for viscosity Model.")
            X_train,y_train = (train_data[:,:-1],train_data[:,-1])
            X_test,y_test = (test_data[:,:-1],test_data[:,-1])

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }


            report = self.evaluate_models(X_train,y_train,X_test,y_test,models)
            
            best_score = max(list(report.values()))

            best_model_name = list(report.keys())[list(report.values()).index(best_score)]

            best_model_obj = models[best_model_name]

            logging.info("Best Model obtained for viscosity")


            save_object(file_path=self.trainer_config.trained_viscosity_model_file_path,obj=best_model_obj)

            return (best_model_name,best_score)
        

        except Exception as e:
            raise CustomException(e,sys)