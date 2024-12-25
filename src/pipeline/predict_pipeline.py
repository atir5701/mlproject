import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipelineDen:
    def __init__(self):
        pass

    def pre(self,features):
        try:
            model_path_density = "assets\densityModel.pkl"
            procesing_path_density = "assets\preprocessorDensity.pkl"
            
            density_object = load_object(model_path_density)
            preprocessing_object = load_object(procesing_path_density)

            data_processed = preprocessing_object.transform(features)
            pred = density_object.predict(data_processed)
            return pred
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomDataDen:
    def __init__(self,MR:float,tk:float,mci:float,wm:float,MW:float):
        self.mr = MR
        self.tk = tk
        self.mci = mci
        self.wm = wm
        self.mw = MW

    def get_data_as_df(self):
        data = {
            'MR' : [self.mr],
            'T/K' : [self.tk],
            'MCI':[self.mci],
            'ωm':[self.wm],
            'MW':[self.mw]
        }
        return pd.DataFrame(data)
    

class CustomDataVis:
    def __init__(self,inv_Temp:float,MCI:float,m:float,omega:float):
        self.inv_temp = inv_Temp
        self.mci = MCI
        self.m = m
        self.omega = omega

    def get_data_as_df(self):
        data = {
            'inv_Temp':[self.inv_temp],
            'MCI':[self.mci],
            'Avg Molar':[self.m], 
            ' omega':[self.omega]
        }
        return pd.DataFrame(data)


class PredictPipelineVis:
    def __init__(self):
        pass

    def pre(self,features):
        try:
            model_path_viscosity = "assets/viscosityModel.pkl"
            procesing_path_viscosity = "assets\preprocessorViscosity.pkl"
            
            density_object = load_object(model_path_viscosity)
            preprocessing_object = load_object(procesing_path_viscosity)

            data_processed = preprocessing_object.transform(features)
            pred = density_object.predict(data_processed)
            return pred
        
        except Exception as e:
            raise CustomException(e,sys)