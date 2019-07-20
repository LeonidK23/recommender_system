import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

class kNNRecommenderII:
    def __init__(self, k):
        self.n_neighbours = k
        self.k_nearest_items = 0
        self.train_data = 0
        self.global_mean = 0

    def fit(self, train_data):
        self.global_mean = np.mean(train_data[:, 2])
        sparse_mat = csr_matrix((train_data[:, 2], (train_data[:, 0], train_data[:, 1])))
        self.train_data = sparse_mat.todense().T
        simil_matrix_items = cosine_similarity(sparse_mat.transpose())
        np.fill_diagonal(simil_matrix_items, -1)
        k_nearest_items_dists = np.sort(simil_matrix_items, axis=1)[:, -self.n_neighbours:]
        self.k_nearest_items = np.array([self.get_neighbour_indices(simil_matrix_items[i, :], k_nearest_items_dists[i, :])
                      for i in range(k_nearest_items_dists.shape[0])])

    def get_neighbour_indices(self, item, distances):
        indices = [np.where(item == dist) for dist in distances]
        return [ind[0][0] for ind in indices]

    def collect_items_user_dict(self, test_data):
        items_to_fill = {}
        for entry in test_data:
            if entry[1] not in items_to_fill:
                items_to_fill[entry[1]] = []
            items_to_fill[entry[1]] += [entry[0].item()]

        return items_to_fill

    def predict(self, test_data):
        items_users_to_fill = self.collect_items_user_dict(test_data)
        predictions = []
        for item in items_users_to_fill:
            neighbours = np.take(self.train_data, self.k_nearest_items[item], axis=0)
            for user in items_users_to_fill[item]:
                user_prediction = int(np.ceil(np.mean(neighbours[:, user])))
                predictions.append([user, item, user_prediction if user_prediction!=0 else
                                    self.global_mean])

        return np.array(predictions)
