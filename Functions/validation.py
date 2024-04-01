from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
import numpy as np

def holdout_validation():
    return "holdout"

def cross_valdiation():
    svc_scores = cross_val_score(svc, selected_features, y_train, cv=10)
    print(svc_scores.mean(), svc_scores.std())

def k_fold_cross_validation(stratified):
    if stratified == True:
        # strat_k_fold_cross validation
        # Stratification was used because it preserves the percentage of samples for each class.
        print("True")
    else:
        # k_fold_cross validation
        print("False")