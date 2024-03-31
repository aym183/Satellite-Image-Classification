from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

def normalise_min_max(train_set, test_set):
    min_max_scaler = MinMaxScaler(feature_range=(-1, 1)) # ------ Doing normalisation before/after pearson has no impact
    x_train_norm = min_max_scaler.fit_transform(train_set)
    x_test_norm = min_max_scaler.fit_transform(test_set)
    return x_train_norm, x_test_norm

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