'''
Contains all the functions required in the feature selection phase
'''

from sklearn.feature_selection import r_regression, f_regression, VarianceThreshold
from sklearn.linear_model import PoissonRegressor
import numpy as np

def calculate_variance_threshold(train_set_x: np.ndarray, test_set_x: np.ndarray, top_10_features: bool) -> tuple:
    '''
    Finds and removes the low variance features from datasets

    Parameters:
    train_set_x: np.ndarray
        The features training dataset
    test_set_x: np.ndarray
        The features testing dataset
    top_10_features: bool
        A boolean indicating whether only the top 10 features should be fetched or not

    Returns:
    top_10_features_train: np.ndarray
        The top 10 features in the training dataset after variance threshold
    top_10_features_test: np.ndarray
        The top 10 features in the testing dataset after variance threshold
    x_train_selected: np.ndarray
        All features in the training dataset after variance threshold
    x_test_selected: np.ndarray
        All features in the testing dataset after variance threshold
    '''
    variances = np.var(train_set_x, axis=0)
    average_variance = np.mean(variances) # To get the threshold value, the average of the variance for each feature was taken - 0.05

    threshold = average_variance
    variance_calculator = VarianceThreshold(threshold)
    variance_calculator.fit(train_set_x)

    if top_10_features:
        kept_features_idx = variance_calculator.get_support(indices=True)
        kept_features_variance = np.var(train_set_x[:, kept_features_idx], axis=0)
        sorted_indices = np.argsort(kept_features_variance)[::-1]
        top_10_features_train = kept_features_idx[sorted_indices][:10]
        top_10_variance_train = kept_features_variance[sorted_indices][:10]

        kept_features_variance_test = np.var(test_set_x[:, kept_features_idx], axis=0)
        sorted_indices_test = np.argsort(kept_features_variance_test)[::-1]
        top_10_features_test = kept_features_idx[sorted_indices_test][:10]
        top_10_variance_test = kept_features_variance_test[sorted_indices_test][:10]

        return top_10_features_train, top_10_features_test
    else: 
        x_train_selected = variance_calculator.transform(train_set_x)
        x_test_selected = variance_calculator.transform(test_set_x)
        return x_train_selected, x_test_selected


def pearson_correlation(train_set_x: np.ndarray, train_set_y: np.ndarray) -> np.ndarray:
    '''
    Calculates the pearson correlation for all the features

    Parameters:
    train_set_x: np.ndarray
        The features training dataset
    train_set_y: np.ndarray
        The classes training dataset

    Returns:
    imp_features: np.ndarray
        A sorted array of the features with the highest correlation
    '''
    pr_coeff = r_regression(train_set_x, train_set_y)
    imp_features = np.argsort(np.abs(pr_coeff))
    return imp_features

def f_regression_scores(train_set_x: np.ndarray, train_set_y: np.ndarray) -> np.ndarray:
    '''
    Calculates the f regression for all the features

    Parameters:
    train_set_x: np.ndarray
        The features training dataset
    train_set_y: np.ndarray
        The classes training dataset

    Returns:
    imp_features: np.ndarray
        A sorted array of the features
    '''
    f_scores, p_value = f_regression(train_set_x, train_set_y)
    imp_features = np.argsort(np.abs(f_scores))
    return imp_features

def poisson_method(train_set_x: np.ndarray, train_set_y: np.ndarray) -> np.ndarray:
    '''
    Performs feature selection using Poisson method

    Parameters:
    train_set_x: np.ndarray
        The features training dataset
    train_set_y: np.ndarray
        The classes training dataset

    Returns:
    top_10_features_idx: np.ndarray
        A sorted array of the features
    '''
    poisson_model = PoissonRegressor()
    poisson_model.fit(train_set_x, train_set_y)
    feature_importance = np.abs(poisson_model.coef_)
    top_10_features_idx = np.argsort(feature_importance)[-10:]
    top_10_features_train = train_set_x[:, top_10_features_idx]

    return top_10_features_idx