from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
import numpy as np

def holdout_cross_validation(svc, x_train, x_test, y_train, y_test):
    svc.fit(x_train, y_train)
    svc.score(x_test, y_test) 

    print("------ Holdout Validation ------")
    print(f"Training Accuracy: {svc.score(x_train, y_train)}")
    print(f"Testing Accuracy: {svc.score(x_test, y_test)}")

def k_fold_cross_valdiation(svc, x_array, y_array, k):
    svc_scores = cross_val_score(svc, x_array, y_array, cv=k)
    cv_mean_accuracy = np.mean(svc_scores)
    print(f"------ {k} fold CV Validation ------")
    print(f"Mean Accuracy: {cv_mean_accuracy}")

def k_fold_cross_validation(stratified):
    if stratified == True:
        # strat_k_fold_cross validation
        # Stratification was used because it preserves the percentage of samples for each class.
        print("True")
    else:
        # k_fold_cross validation
        print("False")