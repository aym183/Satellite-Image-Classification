from sklearn.svm import SVC

def svc_classifier(train_set_x, test_set_x, train_set_y, test_set_y, features, no_of_features):
    svc = SVC(kernel='rbf') # Why poly and how does it work
    selected_features = train_set_x[:, features[:no_of_features]]
    selected_test_features = test_set_x[:, features[:no_of_features]]

    svc.fit(selected_features, train_set_y)
    svc.score(selected_test_features, test_set_y) 

    return f"Training Accuracy: {svc.score(selected_features, train_set_y)}\nTesting Accuracy: {svc.score(selected_test_features, test_set_y)}"