'''
For Metrics
'''
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.neural_network import MLPClassifier

def fetch_classification_report(classifier, test_set_x, test_set_y):
    y_pred = classifier.predict(test_set_x)
    print("----- Classification Report -----")
    print(classification_report(test_set_y, y_pred))

def fetch_accuracy_score(test_set_y, predicted_set_y):
    accuracy = accuracy_score(test_set_y, predicted_set_y)
    print(f"Accuracy: {accuracy}")

def fetch_log_loss(test_set_y, predicted_set_y):
    log_loss = log_loss(test_set_y, predicted_set_y)
    print(f"Log Loss: {log_loss}")