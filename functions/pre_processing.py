'''
Contains all the functions required in pre-processing the datasets 
'''

import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt 
from functions.misc import *

def fetch_datasets() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ''' 
    Fetches all the datasets from the datasets folder

    Returns:
    tuple
        All the loaded datasets
    '''
    x_train = np.load("./datasets/x_train.npy")
    x_test = np.load("./datasets/x_test.npy")
    y_train = np.load("./datasets/y_train.npy")
    y_test = np.load("./datasets/y_test.npy")

    return x_train, x_test, y_train, y_test

def find_missing_values(input_array: np.ndarray) -> dict:
    ''' 
    Finds the missing values (if any) in all the datasets

    Keyword Arguments:
    input_array: np.ndarray
        The dataset values
    array_name: str
        The name of the array

    Returns:
    str
        A string indicating the missing values found in the dataset
    '''
   
    return np.isnan(input_array).sum()

def find_non_unique_features(x_set: np.ndarray, array_name: str) -> list:
    ''' 
    Finds the features with non-unique values (i.e. only one value throughout)

    Keyword Arguments:
    x_set: np.ndarray
        The dataset that contains features
    array_name: str
        The name of the array

    Returns:
    list
        A list containing the features with non unique values
    '''
    # In the test set, there was one feature with 2 unique values, but I chose to keep this as if this feature has strong correlation with the target variablee, it could still be useful
    num_features = x_set.shape[1]
    unique_value_counts = [len(np.unique(x_set[:, i])) for i in range(num_features)]
    filtered_features = [(i+1, count) for i, count in enumerate(unique_value_counts) if count == 1]

    if len(filtered_features) == 0:
        print(f"No features with 1 unique value in {array_name}")
    else:
        for feature, count in filtered_features:
            print("Features with less than 100 unique values:")
            print(f"Feature {feature}: {count} unique values")
    
    return filtered_features

# Assuming numerical features are not categorical
def find_categorical_features(x_set: np.ndarray, array_name: str) -> list:
    ''' 
    Finds the categorical features in a dataset

    Keyword Arguments:
    x_set: np.ndarray
        The dataset that contains features
    array_name: str
        The name of the array

    Returns:
    list
        A list containing the features that have categorical values
        
    '''
    num_columns = x_set.shape[1]
    categorical_features = []

    for i in range(num_columns):
        if not (is_int(x_set[0][i])) and not (is_float(x_set[0][i])):
            categorical_features.append(i)

    if len(categorical_features) == 0:
        print(f"No categorical features in {array_name}")
    else:
        print(f"There are {len(categorical_features)} categorical features")

    # If more than one column exists, encode it?
    return categorical_features