import numpy as np

class RandomRecommender:
    def __init__(self):
        self.min = 0
        self.max = 0

    def fit(self, train_data):
        self.min = np.min(train_data[:, -1])
        self.max = np.max(train_data[:, -1])

    def predict(self, test_data):
        predictions = np.random.randint(self.min, self.max + 1, size=(test_data.shape[0], 1))

        return np.append(test_data, predictions, axis=1)
