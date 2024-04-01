from sklearn.svm import SVC
import numpy as np
from functions.validation import *

def svc_classifier(train_set_x, test_set_x, train_set_y, test_set_y, cross_validation_type):
    svc = SVC(kernel='rbf') # Why poly and how does it work
    selected_features = train_set_x
    selected_test_features = test_set_x

    if cross_validation_type == "holdout":
        holdout_cross_validation(svc, selected_features, selected_test_features, train_set_y, test_set_y)
    elif cross_validation_type == "k_fold":
        concatenated_array_x = np.concatenate((selected_features, selected_test_features), axis=0)
        concatenated_array_y = np.concatenate((train_set_y, test_set_y), axis=0)
        k_fold_cross_valdiation(svc, concatenated_array_x, concatenated_array_y, 10)

    elif cross_validation_type == "k_fold_strat":
        print("Hi")
