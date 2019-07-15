import numpy as np

class MeanRecommender:
    def __init__(self):
        self.mean = 0

    def fit(self, train_data):
        self.mean = np.mean(train_data[:, -1])

    def predict(self, test_data):
        predictions = np.full((test_data.shape[0], 1), self.mean)

        return np.append(test_data, predictions, axis=1)
