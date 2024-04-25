# This class is going to be used to take the dataset from a path
# Abstracts everything so that a user can input a dataset -> all outliers are found and removed, model performed

class DatasetManager:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.get_input()

    def get_input(self):
        print("Please input your program:")
        self.program = input()
        print(self.program)
        print(self.x_train.shape)
        print(self.x_test.shape)
        print(self.y_train.shape)
        print(self.y_test.shape)

