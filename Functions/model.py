import joblib

def save_model(model, file_name):
    joblib.dump(model, file_name)
    print(f"{file_name} has the new model!")

def load_model(file_name):
    return joblib.load(file_name)