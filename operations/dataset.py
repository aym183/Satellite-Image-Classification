import os
import numpy as np
# This class is going to be used to take the dataset from a path
# Abstracts everything so that a user can input a dataset -> all outliers are found and removed, model performed

class DatasetManager:
    def __init__(self):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.get_input()

    def get_input(self):
        print("Please input your program:")
        self.fetch_datasets()
        self.program = input()
        print(self.program)
        print(self.x_train.shape)
        print(self.x_test.shape)
        print(self.y_train.shape)
        print(self.y_test.shape)

    def fetch_datasets(self):
        dataset_folder = "datasets"
        if not os.path.exists(dataset_folder):
            print("Folder not found")
            return

        files = os.listdir(dataset_folder)
        npy_files = [file for file in files if file.endswith(".npy")]
        for file in npy_files:
            filename = f"./datasets/{file}"
            data = np.load(filename)
            if "x_train" in filename:
                self.x_train = data
            elif "x_test" in filename:
                self.x_test = data
            elif "y_train" in filename:
                self.y_train = data
            elif "y_test" in filename:
                self.y_test = data