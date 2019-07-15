import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

class kNNRecommender:
    def __init__(self):
        self.similarity_matrix = 0

    def fit(self, train_data):
        sparse_mat = csr_matrix((train_data[:, 2], (train_data[:, 0], train_data[:, 1])))
        self.similarity_matrix = cosine_similarity(sparse_mat)

    def predict(self, test_data):
        predictions = np.full((test_data.shape[0], 1), self.mean)

        return np.append(test_data, predictions, axis=1)
