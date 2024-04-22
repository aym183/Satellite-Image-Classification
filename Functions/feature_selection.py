from sklearn.feature_selection import r_regression, f_regression, VarianceThreshold
from sklearn.linear_model import PoissonRegressor
import numpy as np

# GPT refer
def calculate_variance_threshold(train_set_x, test_set_x, top_10_features):
    variances = np.var(train_set_x, axis=0)
    average_variance = np.mean(variances) # To get the threshold value, the average of the variance for each feature was taken - 0.05
    # std_dev = np.std(variances) # Due to the high variability of data, standard deviation was taken - 0.026167810885013097
    # print(x_train_selected.shape)
    # print(x_test_selected.shape)

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


def pearson_correlation(train_set_x, train_set_y):
    pr_coeff = r_regression(train_set_x, train_set_y)
    imp_features = np.argsort(np.abs(pr_coeff))
    return imp_features

def f_regression_scores(train_set_x, train_set_y):
    f_scores, p_value = f_regression(train_set_x, train_set_y)
    imp_features = np.argsort(np.abs(f_scores))
    return imp_features

def poisson_method(train_set_x, train_set_y):
    poisson_model = PoissonRegressor()
    poisson_model.fit(train_set_x, train_set_y)
    feature_importance = np.abs(poisson_model.coef_)
    top_10_features_idx = np.argsort(feature_importance)[-10:]
    top_10_features_train = train_set_x[:, top_10_features_idx]

    return top_10_features_idx
