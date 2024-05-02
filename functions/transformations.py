'''
Contains all the functions required in transforming the datasets after pre-processing
'''

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import numpy as np

def normalise_min_max(train_set: np.ndarray, test_set: np.ndarray) -> tuple:
    ''' 
    Normalises the datasets to the range (-1, 1) with MinMaxScaler

    Parameters:
    train_set: np.ndarray
        The training dataset
    test_set: np.ndarray
        The testing dataset

    Returns:
    tuple
        All the normalised datasets
    '''
    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    x_train_norm = min_max_scaler.fit_transform(train_set)
    x_test_norm = min_max_scaler.fit_transform(test_set)
    return x_train_norm, x_test_norm

def normalise_min_max_task_3(train_set: np.ndarray, test_set: np.ndarray) -> tuple:
    ''' 
    Normalises the datasets to the range (10, 15) with MinMaxScaler

    Parameters:
    train_set: np.ndarray
        The training dataset
    test_set: np.ndarray
        The testing dataset

    Returns:
    tuple
        All the normalised datasets
    '''
    min_max_scaler = MinMaxScaler(feature_range=(10,15))
    x_train_norm = min_max_scaler.fit_transform(train_set)
    x_test_norm = min_max_scaler.fit_transform(test_set)
    return x_train_norm, x_test_norm

def standardize_std_scaler(train_set: np.ndarray, test_set: np.ndarray) -> tuple:
    ''' 
    Standardizes the datasets using StandardScaler

    Parameters:
    train_set: np.ndarray
        The training dataset
    test_set: np.ndarray
        The testing dataset

    Returns:
    tuple
        All the standardized datasets
    '''
    standard_scaler = StandardScaler()
    x_train_norm = standard_scaler.fit_transform(train_set)
    x_test_norm = standard_scaler.fit_transform(test_set)
    return x_train_norm, x_test_norm

def normalise_robust_scaler(train_set: np.ndarray, test_set: np.ndarray) -> tuple:
    ''' 
    Scales features with the median and interquartile range using RobustScaler

    Parameters:
    train_set: np.ndarray
        The training dataset
    test_set: np.ndarray
        The testing dataset

    Returns:
    tuple
        All the scaled datasets
    '''
    robust_scaler = RobustScaler()
    x_train_norm = robust_scaler.fit_transform(train_set)
    x_test_norm = robust_scaler.fit_transform(test_set)
    return x_train_norm, x_test_norm

def find_lowest_occurring_class(dataset_y: np.ndarray) -> float:
    ''' 
    Finds the class with the lowest occurrences of values

    Parameters:
    dataset_y: np.ndarray
        The classes dataset

    Returns:
    float
        The class with the lowest occurences
    '''
    unique_classes, class_counts = np.unique(dataset_y, return_counts=True)
    min_count = np.argmin(class_counts)
    lowest_occurrence_class = unique_classes[min_count]
    return lowest_occurrence_class

def find_highest_occurring_class(dataset_y: np.ndarray) -> float:
    ''' 
    Finds the class with the highest occurrences of values

    Parameters:
    dataset_y: np.ndarray
        The classes dataset

    Returns:
    float
        The class with the highest occurences
    '''
    unique_classes, class_counts = np.unique(dataset_y, return_counts=True)
    max_count = np.argmax(class_counts)
    highest_occurrence_class = unique_classes[max_count]
    return highest_occurrence_class

def dataset_undersampling(dataset_x: np.ndarray, dataset_y: np.ndarray) -> tuple:
    ''' 
    Used in imbalanced datasets to reduce the instances in the majority class to balance the class distribution

    Parameters:
    dataset_x: np.ndarray
        The features dataset
    dataset_y: np.ndarray
        The classes dataset

    Returns:
    tuple
        The undersampled versions of the input datasets
    '''
    lowest_occur_class = np.where(dataset_y == find_lowest_occurring_class(dataset_y))[0]
    num_occurences = len(lowest_occur_class)

    selected_indices_other_classes = []
    for class_label in np.unique(dataset_y):
        if class_label != find_lowest_occurring_class(dataset_y):
            class_indices = np.where(dataset_y == class_label)[0]
            selected_indices = np.random.choice(class_indices, size=num_occurences, replace=False)
            selected_indices_other_classes.append(selected_indices)

    balanced_indices = np.concatenate((lowest_occur_class, *selected_indices_other_classes))
    x_balanced = dataset_x[balanced_indices]
    y_balanced = dataset_y[balanced_indices]

    return x_balanced, y_balanced

def dataset_oversampling(dataset_x: np.ndarray, dataset_y: np.ndarray) -> tuple:
    ''' 
    Used in imbalanced datasets to increase the instances in the minority class to balance the class distribution

    Parameters:
    dataset_x: np.ndarray
        The features dataset
    dataset_y: np.ndarray
        The classes dataset

    Returns:
    tuple
        The oversampled versions of the input datasets
    '''
    highest_occur_class = np.where(dataset_y == find_highest_occurring_class(dataset_y))[0]
    num_occurences = len(highest_occur_class)

    selected_indices_other_classes = []
    for class_label in np.unique(dataset_y):
        if class_label != find_highest_occurring_class(dataset_y):
            class_indices = np.where(dataset_y == class_label)[0]
            selected_indices = np.random.choice(class_indices, size=num_occurences)
            selected_indices_other_classes.append(selected_indices)

    balanced_indices = np.concatenate((highest_occur_class, *selected_indices_other_classes))
    x_balanced = dataset_x[balanced_indices]
    y_balanced = dataset_y[balanced_indices]

    return x_balanced, y_balanced

# ------ Task 4 ------

# By reducing to 2, you're getting the 2 most important features
# def reduce_pca_dimensionality(train_set_x, test_set_x, test_set_y):
#     ''' 
#     Finds the features with non-unique values (i.e. only one value throughout)

#     Keyword Arguments:
#     x_set: np.ndarray
#         The dataset that contains features
#     array_name: str
#         The name of the array

#     Returns:
#     list
#         A list containing the features with non unique values
#     '''
#     pca = PCA(n_components=2)
#     x_train_pca = pca.fit_transform(train_set_x)
#     x_test_pca = pca.transform(test_set_x)

#     return x_train_pca, x_test_pca