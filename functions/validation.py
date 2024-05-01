from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, log_loss


'''
---- ADD MORE TESTING METHODS ----
---- MAKE VARS THE SAME ----
'''

def holdout_validation(classifier, x_train, x_test, y_train, y_test):
    classifier.fit(x_train, y_train)

    print("------ Holdout Validation ------")
    print(f"Training Accuracy: {classifier.score(x_train, y_train)}")
    print(f"Testing Accuracy: {classifier.score(x_test, y_test)}")
    return classifier

# USE ONLY TRAINING SET FOR ALL [Test separate]
# CLASSIFICATION REPORT 
def cross_validation(classifier, x_array, y_array):
    cv_score = cross_val_score(classifier, x_array, y_array)
    cv_mean_accuracy = np.mean(cv_score)
    print("------ Cross Validation ------")
    print(f"Mean Accuracy: {cv_mean_accuracy}")
    # print(f"Loss: {classifier.loss_}")
    return classifier

def k_fold_valdiation(x_array, y_array, size, classifier):
    kf = KFold(n_splits=size, shuffle=True) # Mention why shuffling
    tracked_scores = np.zeros(size)
    index = 0
    for train_idx, test_idx in kf.split(x_array):
        x_train_kfold, x_test_kfold = x_array[train_idx], x_array[test_idx]
        y_train_kfold, y_test_kfold = y_array[train_idx], y_array[test_idx]

        if classifier == "svc":
            svc_clf = SVC(kernel='rbf', C=1, gamma="scale")
            svc_clf.fit(x_train_kfold, y_train_kfold)
            tracked_scores[index] = svc_clf.score(x_test_kfold, y_test_kfold)
            index += 1
        else:
            mlp = mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), activation='tanh', solver='adam', alpha=0.05, random_state=42)
            mlp.fit(x_train_kfold, y_train_kfold)
            tracked_scores[index] = mlp.score(x_test_kfold, y_test_kfold)
            index += 1

    print("------ K fold Validation ------")
    print(f"Mean Accuracy: {tracked_scores.mean()}")
    print(f"Std Deviation: {tracked_scores.std()}")
    return classifier
    # print(f"Loss: {classifier.loss_}")

def k_fold_cross_validation_strat(x_array, y_array, size, classifier):
    kf_strat = StratifiedKFold(n_splits=size, shuffle=True) # Mention why shuffling
    tracked_scores = np.zeros(size)
    index = 0
    for train_idx, test_idx in kf_strat.split(x_array, y_array):
        x_train_kfold, x_test_kfold = x_array[train_idx], x_array[test_idx]
        y_train_kfold, y_test_kfold = y_array[train_idx], y_array[test_idx]

        if classifier == "svc":
            svc_clf = SVC(kernel='rbf', C=1, gamma="scale")
            svc_clf.fit(x_train_kfold, y_train_kfold)
            tracked_scores[index] = svc_clf.score(x_test_kfold, y_test_kfold)
            index += 1
        else:
            mlp = mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), activation='tanh', solver='adam', alpha=0.05, random_state=42)
            mlp.fit(x_train_kfold, y_train_kfold)
            tracked_scores[index] = mlp.score(x_test_kfold, y_test_kfold)
            index += 1

    print("------ Stratified K fold Validation ------")
    print(f"Mean Accuracy: {tracked_scores.mean()}")
    print(f"Std Deviation: {tracked_scores.std()}")
    return classifier
    # print(f"Loss: {classifier.loss_}")
    

# ---- Nested CV?