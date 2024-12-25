'''
The role of this code file is to read the data from various source.
The data source can be a cloud sourse, a database.
'''

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass


@dataclass
class DataIngestionConfig:

    trainDataPathDensity = os.path.join("assets","trainden.csv")
    testDataPathDensity = os.path.join("assets","testden.csv")
    rawDataPathDensity = os.path.join("assets","dataden.csv")

    trainDataPathViscosity = os.path.join("assets","trainvis.csv")
    testDataPathViscosity = os.path.join("assets","testdvis.csv")
    rawDataPathViscosity = os.path.join("assets","datavis.csv")




class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Component entered")
        try:
            df1 = pd.read_csv("notebook\data\data_new.csv")
            logging.info("Data Loaded Successfully")
            df2 = pd.read_csv("notebook/data/viscosityData.csv",index_col=0)

            os.makedirs(os.path.dirname(self.config.trainDataPathDensity),exist_ok=True)


            df1.to_csv(self.config.rawDataPathDensity,index=False,header=True)

            train_set,test_set = train_test_split(df1,test_size=0.2,random_state=42)
            train_set.to_csv(self.config.trainDataPathDensity,index=False,header=True)
            test_set.to_csv(self.config.testDataPathDensity,index=False,header=True)

            logging.info("Density Ingestion done successfully")

            df2.to_csv(self.config.rawDataPathViscosity,index=False,header=True)

            train_set,test_set = train_test_split(df2,test_size=0.2,random_state=42)
            train_set.to_csv(self.config.trainDataPathViscosity,index=False,header=True)
            test_set.to_csv(self.config.testDataPathViscosity,index=False,header=True)

            logging.info("Viscosity Ingestion done successfully")

            return ( self.config.trainDataPathDensity,self.config.testDataPathDensity,
                    self.config.trainDataPathViscosity,self.config.testDataPathViscosity)

        except Exception as e:
            
            raise CustomException(e,sys)
        

if __name__ =="__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()