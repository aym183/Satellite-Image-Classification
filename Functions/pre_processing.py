import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt 

def fetch_datasets():
    x_train = np.load("datasets/x_train.npy")
    x_test = np.load("datasets/x_test.npy")
    y_train = np.load("datasets/y_train.npy")
    y_test = np.load("datasets/y_test.npy")

    # print(x_train.shape)
    # print(x_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)

    # All values float

    # print(x_train) 512 features -> Numerical features - Features with values that are continuous on a scale, statistical, or integer-related
    # print(np.unique(y_train)) # [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.] -> 10 classes of possible outputs, and the reoccurences are just one of these values indicating a class 
    return x_train, x_test, y_train, y_test

    
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
    