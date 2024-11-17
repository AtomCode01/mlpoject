import sys
import os
import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransfromationConfig:
    preprocessor_obj_file_path= os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        # making connection to DataTransformationConfig
        self.data_transformation_config = DataTransfromationConfig() 

    def data_transformation_object(self):
        """
        This function transform numerical and categorical data 
        """
        try:
            numerical_columns= ["writing_score","reading_score"]
            categorical_columns= [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # creating num pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("impute",SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            # creating cat_pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("impute",SimpleImputer(strategy="most_frequent" )),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler())
                ]
            )
            logging.info(f"Numerical columns {numerical_columns}")
            logging.info(f"Categorical columns {categorical_columns}")

            # combining both pipelines
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline, numerical_columns),
                    ("cat_pipeline",cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transfromation(self, train_path, test_path):
        try:
            train_df= pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.data_transformation_object()

            target_column_name = "math_score"
            
            # dropping target column form train data
            input_feature_train_df= train_df.drop(columns=[target_column_name], axis=1) # = X_train
            target_feature_train_df = train_df[target_column_name] # = y_train

            input_feature_test_df= test_df.drop(columns=[target_column_name], axis=1) # = X_test
            target_feature_test_df = test_df[target_column_name] # = y_test


            logging.info("Appying preprocessing obj on training dataset and testing dataset")

            input_feature_train_arr= preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr= preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(input_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(input_feature_test_df)]


            # this code saves obj in file path
            save_object (
                file_path= self.data_transformation_config.preprocessor_obj_file_path, # model path
                obj= preprocessing_obj # model
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path, #pickle file path
            )
        except Exception as e:
            raise CustomException(e, sys)