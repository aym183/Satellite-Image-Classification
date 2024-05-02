'''
For Metrics
'''
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import numpy as np
from typing import Union


def fetch_classification_report(classifier: Union[SVC, MLPClassifier], test_set_x: np.ndarray, test_set_y: np.ndarray) -> None:
    '''
    Creates the classification report for a classifier to evaluate the precision, recall, f1-score, and support

    Keyword Arguments:
    classifier: Union[SVC, MLPClassifier]
        The classifier that was trained during development - Only accepts SVC and MLP for now
    test_set_x: np.ndarray
        The features testing dataset
    test_set_y: np.ndarray
        The classes testing dataset
    '''
    y_pred = classifier.predict(test_set_x)
    print("----- Classification Report -----")
    print(classification_report(test_set_y, y_pred))

def fetch_multiple_classification_report(classifiers: Union[SVC, MLPClassifier], classifier_titles: list[str], test_set_x: np.ndarray, test_set_y: np.ndarray):
    '''
    Creates multiple classification reports used for comparison

    Keyword Arguments:
    classifiers: Union[SVC, MLPClassifier]
        The classifier that was trained during development - Only accepts SVC and MLP for now
    classifier_titles: list[str]
        The list of all titles for each report
    test_set_x: np.ndarray
        The features testing dataset
    test_set_y: np.ndarray
        The classes testing dataset
    '''
    for idx in range(len(classifiers)):
        y_pred = classifiers[idx].predict(test_set_x)
        print(f"----- {classifier_titles[idx]} -----\n{classification_report(test_set_y, y_pred)}")

def fetch_accuracy_score(test_set_y: np.ndarray, predicted_set_y: np.ndarray):
    '''
    Details the accuracy of a model based on the actual and predicted values

    Keyword Arguments:
    test_set_y: np.ndarray
        The classes testing dataset
    predicted_set_y: np.ndarray
        The predictions made by the model
    '''
    accuracy = accuracy_score(test_set_y, predicted_set_y)
    print(f"Accuracy: {accuracy}")

def fetch_log_loss(test_set_y: np.ndarray, predicted_set_y: np.ndarray):
    '''
    Details the log loss of a model based on the actual and predicted values

    Keyword Arguments:
    test_set_y: np.ndarray
        The classes testing dataset
    predicted_set_y: np.ndarray
        The predictions made by the model
    '''
    lg_loss = log_loss(test_set_y, predicted_set_y)
    print(f"Log Loss: {lg_loss}")