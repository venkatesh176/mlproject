import os
import sys
from dataclasses import dataclass


import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor
)

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and testing data")

            X_train, y_train,X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "linear Regresiion": LinearRegression(),
                "K-neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=False),
                "AdaBoostRegressor": AdaBoostRegressor(),
            }

            logging.info("models objects created")

            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test = X_test, y_test=y_test, models=models)

            logging.info("model evelauation completed")

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            if best_model_score<0.6:
                logging.info("No best Model Found")

                raise CustomException("No best Model Found")
            
            best_model = models[best_model_name]
            logging.info(f"Best found model on training and testing dataset{best_model_name}")


            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            
            predicted = best_model.predict(X_test)
            r2_s = r2_score(y_test,predicted)

            return r2_s

        except Exception as e:
            raise CustomException(e, sys)


