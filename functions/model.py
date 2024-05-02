'''
This file contains all the functions required for model operations post training
'''
import joblib
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from typing import Union

def save_model(model: Union[SVC, MLPClassifier], file_name: str):
    '''
    Saves the model using joblib

    Parameters:
    model: Union[SVC, MLPClassifier]
        The model that was trained during development - Only accepts SVC and MLP for now
    file_name: str
        The classes training dataset
    '''
    joblib.dump(model, file_name)
    print(f"{file_name} has the new model!")

def load_model(file_name: str):
    '''
    Loads the model using joblib

    Parameters:
    file_name: str
        The path where the model was saved
    
    Returns:
        The loaded model
    '''
    return joblib.load(file_name)

def find_best_configuration_svc(parameters: dict, train_set_x: np.ndarray, test_set_x: np.ndarray, train_set_y: np.ndarray, test_set_y: np.ndarray) -> dict:
    '''
    Performs manual hyperparameter optimisation to find the best configuration for the SVC classifier

    Parameters:
    parameters: dict
        All the parameters to be experimented with
    train_set_x: str
        The features training dataset
    test_set_x: str
        The features testing dataset
    train_set_y: str
        The classes training dataset
    test_set_y: str
        The classes testing dataset
    
    Returns:
        The best parameter configuration with the accuracy
    '''
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
        "testing_accuracy": best_test
    }

def find_best_configuration_mlp(parameters: dict, train_set_x: np.ndarray, test_set_x: np.ndarray, train_set_y: np.ndarray, test_set_y: np.ndarray) -> dict:
    '''
    Performs manual hyperparameter optimisation to find the best configuration for the MLP classifier

    Parameters:
    parameters: dict
        All the parameters to be experimented with
    train_set_x: str
        The features training dataset
    test_set_x: str
        The features testing dataset
    train_set_y: str
        The classes training dataset
    test_set_y: str
        The classes testing dataset
    
    Returns:
        The best parameter configuration with the accuracy
    '''
    selected_hidden_layers = 0 
    selected_activation = ''
    selected_solver = '' 
    selected_alpha = '' 
    best_training = 0
    best_test = 0
    outer_cv = KFold(n_splits=10, shuffle=True)

    for layer in range(len(parameters["hidden_layers"])):
        for activation in range(len(parameters["activations"])):
            for solver in range(len(parameters["solvers"])):
                for alpha in range(len(parameters["alphas"])):
                        inner_scores = []
                        mlp_clf = MLPClassifier(hidden_layer_sizes=parameters["hidden_layers"][layer], activation=parameters["activations"][activation], solver=parameters["solvers"][solver], alpha=parameters["alphas"][alpha])
                        mlp_clf.fit(train_set_x, train_set_y)
                        current_train = mlp_clf.score(train_set_x, train_set_y)
                        current_test = mlp_clf.score(test_set_x, test_set_y)
                        # Nested CV
                        for train_index, val_index in outer_cv.split(train_set_x):
                            X_train, X_val = train_set_x[train_index], train_set_x[val_index]
                            Y_train, Y_val = train_set_y[train_index], train_set_y[val_index]
                            mlp_clf.fit(X_train, Y_train)
                            test_score = mlp_clf.fit(X_val, Y_val)
                            inner_scores.append(test_score)
                        
                        mean_score = sum(inner_scores) / len(inner_scores)
                        if (current_train > best_training) and (current_test > best_test):
                            best_training = current_train
                            best_test = current_test
                            selected_hidden_layers = parameters["hidden_layers"][layer]
                            selected_activation = parameters["activations"][activation]
                            selected_solver = parameters["solvers"][solver]
                            selected_alpha = parameters["alphas"][alpha]

    return {
        "parameters": {"hidden_layers": selected_hidden_layers, "activation": selected_activation, "solver": selected_solver, "alpha": selected_alpha},
        "training_accuracy": best_training,
        "testing_accuracy": best_test
    }

                        # print(f"------- With hidden_layers={hidden_layers[layer]}, activation={activations[activation]}, solver={solvers[solver]}, alpha={alphas[alpha]}")
                        # holdout_validation(mlp_clf, x_train_norm, x_test_norm, y_train, y_test)
                        # print(f"Mean - {mean_score}")