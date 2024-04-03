'''
For Metrics
'''
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

def fetch_classification_report(classifier, test_set_x, test_set_y):
    y_pred = classifier.predict(test_set_x)
    print("----- Classification Report -----")
    print(classification_report(test_set_y, y_pred))