import os 
import sys 
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging 

from src.utils import save_object,evaluate_model 

@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()



    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split the training and test input data")
            xTrain,yTrain,xTest,yTest = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=0),
                "AdaBoost": AdaBoostRegressor(),
                "K-Nearest Neighbors": KNeighborsRegressor()
                }
            model_report:dict = evaluate_model(xTrain = xTrain , yTrain  = yTrain ,xTest = xTest , yTest = yTest, models = models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            if best_model_score < 0.6 :
                raise CustomException("No best Model fount")
            logging.info("Best found model on both training and testing dataset")

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(xTest)

            r2_square = r2_score(yTest,predicted)
            return r2_square

        except:
            pass
