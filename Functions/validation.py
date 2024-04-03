from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.svm import SVC
import numpy as np

def holdout_validation(svc, x_train, x_test, y_train, y_test):
    svc.fit(x_train, y_train)
    svc.score(x_test, y_test) 

    print("------ Holdout Validation ------")
    print(f"Training Accuracy: {svc.score(x_train, y_train)}")
    print(f"Testing Accuracy: {svc.score(x_test, y_test)}")

# USE ONLY TRAINING SET FOR ALL [Test separate]
# CLASSIFICATION REPORT 
def cross_validation(svc, x_array, y_array, size):
    cv_score = cross_val_score(svc, x_array, y_array, cv=size)
    cv_mean_accuracy = np.mean(cv_score)
    print(f"------ Cross Validation ------")
    print(f"Mean Accuracy: {cv_mean_accuracy}")

def k_fold_valdiation(x_array, y_array, size):
    kf = KFold(n_splits=size)
    tracked_scores = np.zeros(size)
    index = 0
    for train_idx, test_idx in kf.split(x_array):
        x_train_kfold, x_test_kfold = x_array[train_idx], x_array[test_idx]
        y_train_kfold, y_test_kfold = y_array[train_idx], y_array[test_idx]

        svc_clf = SVC()
        svc_clf.fit(x_train_kfold, y_train_kfold)
        tracked_scores[index] = svc_clf.score(x_test_kfold, y_test_kfold)
        index += 1

    print(f"------ K fold Validation ------")
    print(f"Mean Accuracy: {tracked_scores.mean()}")
    print(f"Std Deviation: {tracked_scores.std()}")

def k_fold_cross_validation_strat(x_array, y_array, size):
    kf_strat = StratifiedKFold(n_splits=size)
    tracked_scores = np.zeros(size)
    index = 0
    for train_idx, test_idx in kf_strat.split(x_array, y_array):
        x_train_kfold, x_test_kfold = x_array[train_idx], x_array[test_idx]
        y_train_kfold, y_test_kfold = y_array[train_idx], y_array[test_idx]

        svc_clf = SVC()
        svc_clf.fit(x_train_kfold, y_train_kfold)
        tracked_scores[index] = svc_clf.score(x_test_kfold, y_test_kfold)
        index += 1

    print(f"------ Stratified K fold Validation ------")
    print(f"Mean Accuracy: {tracked_scores.mean()}")
    print(f"Std Deviation: {tracked_scores.std()}")
    

# ---- Nested CV?