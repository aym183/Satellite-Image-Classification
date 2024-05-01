from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import numpy as np
from functions.validation import *
from sklearn.metrics import accuracy_score, classification_report

# Configuration change all from hyperparameter

def svc_classifier(train_set_x, test_set_x, train_set_y, test_set_y,):
    svc = SVC(kernel='rbf', C=1, gamma="scale", probability=True) # Configuration derived from hyperparameter tuning
    selected_features = train_set_x
    selected_test_features = test_set_x
    # concatenated_array_x = np.concatenate((selected_features, selected_test_features), axis=0)
    # concatenated_array_y = np.concatenate((train_set_y, test_set_y), axis=0)
    holdout_validation(svc, selected_features, selected_test_features, train_set_y, test_set_y)
    cross_validation(svc, train_set_x, train_set_y)
    k_fold_valdiation(train_set_x, train_set_y, 20, "svc")
    k_fold_cross_validation_strat(train_set_x, train_set_y, 10, "svc")

    return svc

def mlp_classifier(train_set_x, test_set_x, train_set_y, test_set_y):
    mlp = MLPClassifier(hidden_layer_sizes=(50, 100, 50), activation='tanh', solver='sgd', alpha=0.0001, random_state=42, shuffle=False) # Configuration derived from hyperparameter tuning
    mlp.fit(train_set_x, train_set_y)
    # concatenated_array_x = np.concatenate((train_set_x, test_set_x), axis=0)
    # concatenated_array_y = np.concatenate((train_set_y, test_set_y), axis=0)
    holdout_validation(mlp, train_set_x, test_set_x, train_set_y, test_set_y)
    cross_validation(mlp, train_set_x, train_set_y)
    k_fold_valdiation(train_set_x, train_set_y, 20, "mlp")
    k_fold_cross_validation_strat(train_set_x, train_set_y, 10, "mlp")

    return mlp