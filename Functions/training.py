from sklearn.svm import SVC

def train_svc_10_features(train_set_x, test_set_x, train_set_y, test_set_y, features):
    svc = SVC(kernel='rbf') # Why poly and how does it work
    selected_features = train_set_x[:, features[:10]]
    selected_test_features = test_set_x[:, features[:10]]

    svc.fit(selected_features, train_set_y)
    svc.score(selected_test_features, test_set_y) 

    return f"Training Accuracy: {svc.score(selected_features, train_set_y)}\nTesting Accuracy: {svc.score(selected_test_features, test_set_y)}"