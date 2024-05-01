import joblib
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def save_model(model, file_name):
    joblib.dump(model, file_name)
    print(f"{file_name} has the new model!")

def load_model(file_name):
    return joblib.load(file_name)

def find_best_configuration_svc(parameters, train_set_x, test_set_x, train_set_y, test_set_y):
    selected_c = 0 
    selected_kernel = ''
    selected_gamma = '' 
    best_training = 0
    best_test = 0

    for values in range(len(parameters["kernels_values"])):
     for c_vals in range(len(parameters["c_values"])):
          for g_values in range(len(parameters["gamma_values"])):
                svc_clf = SVC(C=parameters["c_values"][c_vals], gamma=parameters["gamma_values"][g_values], kernel=parameters["kernels_values"][values])
                svc_clf.fit(train_set_x, train_set_y)
                current_train = svc_clf.score(train_set_x, train_set_y) 
                current_test = svc_clf.score(test_set_x, test_set_y)
                cv_score = cross_val_score(svc_clf, train_set_x, train_set_y, cv=10)

                if (current_train > best_training) and (current_test > best_test):
                    best_training = current_train
                    best_test = current_test
                    selected_c = parameters["c_values"][c_vals]
                    selected_kernel = parameters["kernels_values"][values]
                    selected_gamma = parameters["gamma_values"][g_values]

    return {
        "parameters": {"c": selected_c, "kernel": selected_kernel, "gamma": selected_gamma},
        "training_accuracy": best_training,
        "testing_accuracy": best_test,
    }

def find_best_configuration_mlp(parameters, train_set_x, test_set_x, train_set_y, test_set_y):
    print("MLP")