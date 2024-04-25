import os
import numpy as np
import joblib
from typing import List, Dict
# This class is going to be used to take the dataset from a path
# Abstracts everything so that a user can input a dataset -> all outliers are found and removed, model performed

class DatasetManager:
    def __init__(self):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.fetch_datasets()

    '''
    --------- FETCHING DATASETS ---------
    '''
    def fetch_datasets(self):
        print("Checking for datasets") # Check for saved datasets first
        # Add something to check if saved available -> Show user if want to use those
        # Assign those to the attributes

        dataset_folder = "datasets"
        if not os.path.exists(dataset_folder):
            print("Folder not found")
            return

        files = os.listdir(dataset_folder) 
        npy_files = [file for file in files if file.endswith(".npy")] # ADD ERROR CHECKING WHEN FILES DONT EXIST
        for file in npy_files:
            filename = f"./datasets/{file}"
            data = np.load(filename)
            if "x_train" in filename:
                self.x_train = data
            elif "x_test" in filename:
                self.x_test = data
            elif "y_train" in filename:
                self.y_train = data
            elif "y_test" in filename:
                self.y_test = data
        print("datasets found!")
        print(f"x_train -> {self.x_train.shape}")
        print(f"y_train -> {self.x_test.shape}")
        print(f"x_test -> {self.y_train.shape}")
        print(f"y_test -> {self.y_test.shape}")

    def save_dataset(self, dataset, file_name):
        joblib.dump(dataset, file_name)
        print(f"{file_name} has the new dataset!")
    
    def load_dataset(self, dataset):
        return joblib.load(dataset)
    
    '''
    --------- OUTLIER DETECTION ---------
    '''
    def outlier_detection(self):
        print("***** Detecting for outliers *****")
        keys_with_missing_values = self.find_missing_values()
        if keys_with_missing_values:
            print(f"{', '.join(keys_with_missing_values)} has missing values\nHow would you like to remediate this? Options - (i) Imputer, etc, etc")
            missing_vals_input = input()
            print(f"{missing_vals_input}")
            
            

# , input: list, array_name: str
    def find_missing_values(self) -> List[str]: 
        missing_values = {}
        for key, value in vars(self).items():
            missing_values_count = np.isnan(value).sum()
            missing_values[key] = missing_values_count != 0

        keys_with_missing_values = [key for key, value in missing_values.items() if value]
        if not keys_with_missing_values:
            print("No missing values found")
            
        return keys_with_missing_values

    '''
    --------- OUTLIER REMEDIATION ---------
    '''

    '''
    --------- FIND CATEGORICAL FEATURES ---------
    '''

    '''
    --------- CATEGORICAL FEATURES REMEDIATION ---------
    '''

    '''
    --------- TRANSFORMATION ---------
    '''