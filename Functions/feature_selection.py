from sklearn.feature_selection import r_regression
import numpy as np

def pearson_correlation(train_set_x, train_set_y):
    pr_coeff = r_regression(train_set_x, train_set_y)
    imp_features = np.argsort(np.abs(pr_coeff))
    return imp_features
