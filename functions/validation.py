'''
This file contains all the functions used for validating the models
'''
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, log_loss
from typing import Union

def holdout_validation(classifier: Union[MLPClassifier, SVC], train_set_x: np.ndarray, test_set_x: np.ndarray, train_set_y: np.ndarray, test_set_y: np.ndarray) -> Union[MLPClassifier|SVC]:
    '''
    Performs holdout validation on the classifier used for training

    Parameters:
    classifier: Union[SVC, MLPClassifier]
        The model that was trained during development - Only accepts SVC and MLP for now
    train_set_x: np.ndarray
        The features training dataset
    test_set_x: np.ndarray
        The features testing dataset
    train_set_y: np.ndarray
        The classes training dataset
    test_set_y: np.ndarray
        The classes testing dataset
    
    Returns:
    classifier: Union[SVC, MLPClassifier]
        The classifer that was validated
    '''
    classifier.fit(train_set_x, train_set_y)
    print("------ Holdout Validation ------")
    print(f"Training Accuracy: {classifier.score(train_set_x, train_set_y)}")
    print(f"Testing Accuracy: {classifier.score(test_set_x, test_set_y)}")
    return classifier

def cross_validation(classifier: Union[MLPClassifier, SVC], train_set_x: np.ndarray, train_set_y: np.ndarray) -> Union[MLPClassifier, SVC]:
    '''
    Performs cross validation on the classifier used for training

    Parameters:
    classifier: Union[SVC, MLPClassifier]
        The model that was trained during development - Only accepts SVC and MLP for now
    train_set_x: np.ndarray
        The features training dataset
    train_set_y: np.ndarray
        The classes training dataset
    
    Returns:
    classifier: Union[SVC, MLPClassifier]
        The classifer that was validated
    '''
    cv_score = cross_val_score(classifier, train_set_x, train_set_y)
    cv_mean_accuracy = np.mean(cv_score)
    print("------ Cross Validation ------")
    print(f"Mean Accuracy: {cv_mean_accuracy}")
    return classifier

def k_fold_valdiation(classifier: Union[MLPClassifier, SVC], train_set_x: np.ndarray, train_set_y: np.ndarray, size: int) -> Union[MLPClassifier, SVC]:
    '''
    Performs K-fold validation on the classifier used for training

    Parameters:
    classifier: Union[SVC, MLPClassifier]
        The model that was trained during development - Only accepts SVC and MLP for now
    train_set_x: np.ndarray
        The features training dataset
    train_set_y: np.ndarray
        The classes training dataset
    size: int
        The number of folds
    
    Returns:
    classifier: Union[SVC, MLPClassifier]
        The classifer that was validated
    '''
    kf = KFold(n_splits=size, shuffle=True)
    tracked_scores = np.zeros(size)
    index = 0
    for train_idx, test_idx in kf.split(train_set_x):
        x_train_kfold, x_test_kfold = train_set_x[train_idx], train_set_x[test_idx]
        y_train_kfold, y_test_kfold = train_set_y[train_idx], train_set_y[test_idx]

        if classifier == "svc":
            # kernel='rbf', C=1, gamma="scale"
            svc_clf = SVC()
            svc_clf.fit(x_train_kfold, y_train_kfold)
            tracked_scores[index] = svc_clf.score(x_test_kfold, y_test_kfold)
            index += 1
        else:
            # hidden_layer_sizes=(100,), activation='tanh', solver='adam', alpha=0.05, random_state=42)
            mlp = MLPClassifier()
            mlp.fit(x_train_kfold, y_train_kfold)
            tracked_scores[index] = mlp.score(x_test_kfold, y_test_kfold)
            index += 1

    print("------ K fold Validation ------")
    print(f"Mean Accuracy: {tracked_scores.mean()}")
    print(f"Std Deviation: {tracked_scores.std()}")
    return classifier

def k_fold_cross_validation_strat(classifier: Union[MLPClassifier, SVC], train_set_x: np.ndarray, train_set_y: np.ndarray, size: int) -> Union[MLPClassifier, SVC]:
    '''
    Performs stratified K-fold validation on the classifier used for training

    Parameters:
    classifier: Union[SVC, MLPClassifier]
        The model that was trained during development - Only accepts SVC and MLP for now
    train_set_x: np.ndarray
        The features training dataset
    train_set_y: np.ndarray
        The classes training dataset
    size: int
        The number of folds
    
    Returns:
    classifier: Union[SVC, MLPClassifier]
        The classifer that was validated
    '''    
    kf_strat = StratifiedKFold(n_splits=size, shuffle=True)
    tracked_scores = np.zeros(size)
    index = 0
    for train_idx, test_idx in kf_strat.split(train_set_x, train_set_y):
        x_train_kfold, x_test_kfold = train_set_x[train_idx], train_set_x[test_idx]
        y_train_kfold, y_test_kfold = train_set_y[train_idx], train_set_y[test_idx]

        if classifier == "svc":
            # kernel='rbf', C=1, gamma="scale"
            svc_clf = SVC()
            svc_clf.fit(x_train_kfold, y_train_kfold)
            tracked_scores[index] = svc_clf.score(x_test_kfold, y_test_kfold)
            index += 1
        else:
            # hidden_layer_sizes=(100,), activation='tanh', solver='adam', alpha=0.05, random_state=42
            mlp = MLPClassifier()
            mlp.fit(x_train_kfold, y_train_kfold)
            tracked_scores[index] = mlp.score(x_test_kfold, y_test_kfold)
            index += 1

    print("------ Stratified K fold Validation ------")
    print(f"Mean Accuracy: {tracked_scores.mean()}")
    print(f"Std Deviation: {tracked_scores.std()}")
    return classifier