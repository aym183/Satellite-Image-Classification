'''
This file contains all the functions required in training each model
'''

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import numpy as np
from functions.validation import *
from sklearn.metrics import accuracy_score, classification_report

def svc_classifier(train_set_x: np.ndarray, test_set_x: np.ndarray, train_set_y: np.ndarray, test_set_y: np.ndarray) -> SVC:
    ''' 
    Trains the model under a Support Vector Classifier

    Parameters:
    train_set_x: np.ndarray
        The features training dataset
    train_set_y: np.ndarray
        The classes training dataset
    test_set_x: np.ndarray
        The features testing dataset
    test_set_y: np.ndarray
        The classes testing dataset

    Returns:
    svc: SVC
        The trained model
    '''
    # kernel='rbf', C=1, gamma="scale", probability=True
    svc = SVC(probability=True) # Configuration derived from hyperparameter tuning
    selected_features = train_set_x
    selected_test_features = test_set_x
    # concatenated_array_x = np.concatenate((selected_features, selected_test_features), axis=0)
    # concatenated_array_y = np.concatenate((train_set_y, test_set_y), axis=0)
    holdout_validation(svc, selected_features, selected_test_features, train_set_y, test_set_y)
    cross_validation(svc, train_set_x, train_set_y)
    k_fold_valdiation("svc", train_set_x, train_set_y, 20)
    k_fold_cross_validation_strat("svc", train_set_x, train_set_y, 10)

    return svc

def mlp_classifier(train_set_x: np.ndarray, test_set_x: np.ndarray, train_set_y: np.ndarray, test_set_y: np.ndarray) -> MLPClassifier:
    ''' 
    Trains the model under a Multi Layer Perceptron Classifier

    Parameters:
    train_set_x: np.ndarray
        The features training dataset
    train_set_y: np.ndarray
        The classes training dataset
    test_set_x: np.ndarray
        The features testing dataset
    test_set_y: np.ndarray
        The classes testing dataset

    Returns:
    mlp: MLPClassifier
        The trained model
    '''
    # hidden_layer_sizes=(50, 100, 50), activation='tanh', solver='sgd', alpha=0.0001, random_state=42, shuffle=False
    mlp = MLPClassifier() # Configuration derived from hyperparameter tuning
    mlp.fit(train_set_x, train_set_y)
    # concatenated_array_x = np.concatenate((train_set_x, test_set_x), axis=0)
    # concatenated_array_y = np.concatenate((train_set_y, test_set_y), axis=0)
    holdout_validation(mlp, train_set_x, test_set_x, train_set_y, test_set_y)
    cross_validation(mlp, train_set_x, train_set_y)
    k_fold_valdiation(train_set_x, train_set_y, 20, "mlp")
    k_fold_cross_validation_strat(train_set_x, train_set_y, 10, "mlp")

    return mlp