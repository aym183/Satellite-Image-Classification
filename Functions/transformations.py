from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import numpy as np

# Normalisation
def normalise_min_max(train_set, test_set):
    min_max_scaler = MinMaxScaler(feature_range=(-1, 1)) # ------ Doing normalisation before/after pearson has no impact
    x_train_norm = min_max_scaler.fit_transform(train_set)
    x_test_norm = min_max_scaler.fit_transform(test_set)
    return x_train_norm, x_test_norm

def normalise_min_max_task_3(train_set, test_set):
    min_max_scaler = MinMaxScaler(feature_range=(10,15)) # ------ Doing normalisation before/after pearson has no impact
    x_train_norm = min_max_scaler.fit_transform(train_set)
    x_test_norm = min_max_scaler.fit_transform(test_set)
    return x_train_norm, x_test_norm

# Standardisation
def normalise_std_scaler(train_set, test_set):
    standard_scaler = StandardScaler()
    x_train_norm = standard_scaler.fit_transform(train_set)
    x_test_norm = standard_scaler.fit_transform(test_set)
    return x_train_norm, x_test_norm

def normalise_robust_scaler(train_set, test_set):
    robust_scaler = RobustScaler()
    x_train_norm = robust_scaler.fit_transform(train_set)
    x_test_norm = robust_scaler.fit_transform(test_set)
    return x_train_norm, x_test_norm

def dataset_undersampling(dataset_x, dataset_y):
    occurrences_of_5 = np.where(dataset_y == 5)[0] # Lowest occurences class
    num_occurences = len(occurrences_of_5)

    selected_indices_other_classes = []
    for class_label in np.unique(dataset_y):
        if class_label != 5:
            class_indices = np.where(dataset_y == class_label)[0]
            selected_indices = np.random.choice(class_indices, size=num_occurences, replace=False)
            selected_indices_other_classes.append(selected_indices)

    balanced_indices = np.concatenate((occurrences_of_5, *selected_indices_other_classes))
    x_balanced = dataset_x[balanced_indices]
    y_balanced = dataset_y[balanced_indices]

    return x_balanced, y_balanced

# First find out in each dataset the most occurring class
def dataset_oversampling(dataset_x, dataset_y):
    occurrences_of_7 = np.where(dataset_y == 7)[0] # Highest occurences class
    num_occurences = len(occurrences_of_7)

    selected_indices_other_classes = []
    for class_label in np.unique(dataset_y):
        if class_label != 7:
            class_indices = np.where(dataset_y == class_label)[0]
            selected_indices = np.random.choice(class_indices, size=num_occurences)
            selected_indices_other_classes.append(selected_indices)

    balanced_indices = np.concatenate((occurrences_of_7, *selected_indices_other_classes))
    x_balanced = dataset_x[balanced_indices]
    y_balanced = dataset_y[balanced_indices]

    return x_balanced, y_balanced