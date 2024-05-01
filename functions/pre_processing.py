import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt 
from functions.misc import *

def fetch_datasets():
    x_train = np.load("./datasets/x_train.npy")
    x_test = np.load("./datasets/x_test.npy")
    y_train = np.load("./datasets/y_train.npy")
    y_test = np.load("./datasets/y_test.npy")

    return x_train, x_test, y_train, y_test

def find_missing_values(input: np.ndarray, array_name: str) -> str:
    missing_values_count = np.isnan(input).sum()
    return f"Missing Values in {array_name} -> {missing_values_count}"

def find_non_unique_features(x_set, array_name):
    # Checking for unique values of features (Should find those that have 1), there were 0 features in the train set with more than 1 unique value
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

# Assuming numerical features are not categorical
def find_categorical_features(train_set):
    num_columns = train_set.shape[1]
    categorical_features = []
    for i in range(num_columns):
        if not (is_int(train_set[0][i])) and not (is_float(train_set[0][i])):
            print(f"Column {i+1} -> {train_set[0][i]}")
            categorical_features.append(i)
    return categorical_features

def plot_feature_split_of_values(input_arrays, input_text):
    fig, axs = plt.subplots(1, len(input_arrays), figsize=(15, 4)) 
    fig.suptitle('Split of Values', fontsize=16)

    for i, input_array in enumerate(input_arrays):
        flattened_array = input_array.flatten()
        axs[i].hist(flattened_array, bins=50, color='blue', alpha=0.7)
        axs[i].set_title(f'{input_text[i]}')
        axs[i].set_xlabel('Values')
        axs[i].set_ylabel('Frequency')
        axs[i].set_xlim(0, 3)
        axs[i].set_xticks(np.arange(0, 3.1, 0.25))

    plt.tight_layout()
    plt.show()

def plot_class_split_of_values(input_arrays, input_text):
    fig, axs = plt.subplots(1, len(input_arrays), figsize=(15, 4)) 
    fig.suptitle('Split of Values', fontsize=16)

    for i, input_array in enumerate(input_arrays):
        flattened_array = input_array.flatten()
        axs[i].hist(flattened_array, bins=50, color='blue', alpha=0.7)
        axs[i].set_title(f'{input_text[i]}')
        axs[i].set_xlabel('Values')
        axs[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()