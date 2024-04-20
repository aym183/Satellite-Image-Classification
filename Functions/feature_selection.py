from sklearn.feature_selection import r_regression, f_regression
import numpy as np

def pearson_correlation(train_set_x, train_set_y):
    pr_coeff = r_regression(train_set_x, train_set_y)
    imp_features = np.argsort(np.abs(pr_coeff))
    return imp_features

def f_regression_scores(train_set_x, train_set_y):
    f_scores, p_value = f_regression(train_set_x, train_set_y)
    imp_features = np.argsort(np.abs(f_scores))
    return imp_features