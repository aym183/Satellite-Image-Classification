from sklearn.feature_selection import r_regression, f_regression, VarianceThreshold
import numpy as np

def calculate_variance_threshold(train_set_x, test_set_x):
    variances = np.var(train_set_x, axis=0)
    # average_variance = np.mean(variances)
    std_dev = np.std(variances) # Due to the high variability of data, standard deviation was taken - 0.026167810885013097

    threshold = std_dev
    variance_calculator = VarianceThreshold(threshold)
    variance_calculator.fit(train_set_x)
    x_train_selected = variance_calculator.transform(train_set_x)
    x_test_selected = variance_calculator.transform(test_set_x)

    print(x_train_selected.shape)
    print(x_test_selected.shape)
    return x_train_selected, x_test_selected


def pearson_correlation(train_set_x, train_set_y):
    pr_coeff = r_regression(train_set_x, train_set_y)
    imp_features = np.argsort(np.abs(pr_coeff))
    return imp_features

def f_regression_scores(train_set_x, train_set_y):
    f_scores, p_value = f_regression(train_set_x, train_set_y)
    imp_features = np.argsort(np.abs(f_scores))
    return imp_features