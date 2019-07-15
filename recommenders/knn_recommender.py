import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

class kNNRecommender:
    def __init__(self, k):
        self.n_neighbours = k
        self.similarity_matrix = 0

    def fit(self, train_data):
        sparse_mat = csr_matrix((train_data[:, 2], (train_data[:, 0], train_data[:, 1])))
        self.similarity_matrix = cosine_similarity(sparse_mat)

    def predict(self, test_data):
        np.fill_diagonal(self.similarity_matrix, -1)
        temp = np.sort(self.similarity_matrix, axis=1)[:, -self.n_neighbours:]
        a = temp[2, :]
        print([np.where(self.similarity_matrix[2, :] == dist) for dist in a])
