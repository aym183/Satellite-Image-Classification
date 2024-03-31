import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt 

def find_missing_values(input: np.ndarray, array_name: str) -> str:
    missing_values_count = np.isnan(input).sum()
    return f"Missing Values in {array_name} -> {missing_values_count}"

def find_negative_values(input: np.ndarray, array_name: str) -> str:
    negative_values_count = (input < 0).sum() 
    return f"Negative Values in {array_name} -> {negative_values_count}"
    
# Reference GPT
def plot_split_of_values(input_arrays):
    fig, axs = plt.subplots(1, len(input_arrays), figsize=(15, 4)) 
    fig.suptitle('Split of Values', fontsize=16)
    
    for i, input_array in enumerate(input_arrays):
        flattened_array = input_array.flatten()
        axs[i].hist(flattened_array, bins=50, color='blue', alpha=0.7)
        axs[i].set_title(f'Array {i+1}')
        axs[i].set_xlabel('Values')
        axs[i].set_ylabel('Frequency')
        axs[i].grid(True)
    
    plt.tight_layout()
    plt.show()
    