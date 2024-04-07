'''
For Metrics
'''
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.neural_network import MLPClassifier

def fetch_classification_report(classifier, test_set_x, test_set_y):
    y_pred = classifier.predict(test_set_x)
    print("----- Classification Report -----")
    print(classification_report(test_set_y, y_pred))

def fetch_multiple_classification_report(classifiers, classifier_titles, test_set_x, test_set_y):
    classification_reports = []
    for idx in range(len(classifiers)):
        y_pred = classifiers[idx].predict(test_set_x)
        print(f"----- {classifier_titles[idx]} -----\n{classification_report(test_set_y, y_pred)}")

def fetch_accuracy_score(test_set_y, predicted_set_y):
    accuracy = accuracy_score(test_set_y, predicted_set_y)
    print(f"Accuracy: {accuracy}")

def fetch_log_loss(test_set_y, predicted_set_y):
    lg_loss = log_loss(test_set_y, predicted_set_y)
    print(f"Log Loss: {lg_loss}")