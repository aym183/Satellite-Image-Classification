import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt 

def find_missing_values():
    missing_values_count = np.isnan(y_train).sum()
    return f"Missing Values is -> {missing_values_count}"

def find_negative_values():
    negative_values_count = (y_train < 0).sum() 
    return f"Negative Values is -> {negative_values_count}"
    
def plot_split_of_values(input_array):
    flattened_array = input_array.flatten()
    plt.figure(figsize=(10, 6))
    plt.hist(flattened_array, bins=50, color='blue', alpha=0.7)
    plt.title('Histogram of x_train Values')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    