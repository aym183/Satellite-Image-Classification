'''
This file contains all the functions required for miscellaneous use-cases
'''
import joblib
import os
import numpy as np

def is_float(string):
    '''
    Check if a string can be converted to float
    '''
    try:
        float(string)
        return True
    except ValueError:
        return False

def is_int(string):
    '''
    Check if a string can be converted to int
    '''
    try:
        int(string)
        return True
    except ValueError:
        return False
    
def save_dataset(model, file_name):
    joblib.dump(model, file_name)
    print(f"{file_name} has the new dataset!")
    
def load_dataset(file_name):
    return joblib.load(file_name)
