import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

class kNNRecommenderUI:
    def __init__(self, k):
        self.n_neighbours = k
        self.k_nearest_users = 0
        self.k_nearest_items = 0
        self.train_data = 0
        self.global_mean = 0

    def fit(self, train_data):
        self.global_mean = np.mean(train_data[:, 2])
        sparse_mat = csr_matrix((train_data[:, 2], (train_data[:, 0], train_data[:, 1])))
        self.train_data = sparse_mat
        simil_matrix_users = cosine_similarity(sparse_mat)
        np.fill_diagonal(simil_matrix_users, -1)
        k_nearest_users_dists = np.sort(simil_matrix_users, axis=1)[:, -self.n_neighbours:]
        self.k_nearest_users = np.array([self.get_neighbour_indices(simil_matrix_users[i, :], k_nearest_users_dists[i, :]) + [i]
                      for i in range(k_nearest_users_dists.shape[0])])
        # Find nearest neighbours for each item
        simil_matrix_items = cosine_similarity(sparse_mat.transpose())
        np.fill_diagonal(simil_matrix_items, -1)
        k_nearest_items_dists = np.sort(simil_matrix_items, axis=1)[:, -self.n_neighbours:]
        self.k_nearest_items = np.array([self.get_neighbour_indices(simil_matrix_items[i, :], k_nearest_items_dists[i, :]) + [i]
                      for i in range(k_nearest_items_dists.shape[0])])

    def get_neighbour_indices(self, user, distances):
        indices = [np.where(user == dist) for dist in distances]
        return [ind[0][0] for ind in indices]

    def collect_user_items_dict(self, test_data):
        items_to_fill = {}
        for entry in test_data:
            if entry[0] not in items_to_fill:
                items_to_fill[entry[0]] = []
            items_to_fill[entry[0]] += [entry[1].item()]

        return items_to_fill

    def predict(self, test_data):
        users_items_to_fill = self.collect_user_items_dict(test_data)
        predictions = []
        for user in users_items_to_fill:
            for item in users_items_to_fill[user]:
                # neighbours_users = np.take(self.train_data, self.k_nearest_users[user], axis=0) # old version
                # neighbours_users = self.train_data[np.ix_(self.k_nearest_users[user], self.k_nearest_items[item])] # with dense matrix
                neighbours_users = self.train_data[np.ix_(self.k_nearest_users[user], self.k_nearest_items[item])].data
                # select from training data nearest users and nearest items to the unknow item/user
                # except the unknown value of item to fill
                # item_prediction = np.mean(np.asarray(neighbours_users).flatten()[:-1]) # try this with dense matrix
                item_prediction = self.global_mean if neighbours_users.shape[0] == 0 else np.mean(neighbours_users)
                # item_prediction = int(np.ceil(nearest_users_items.mean())) # Try: not round or use floor
                predictions.append([user, item, item_prediction])

        return np.array(predictions)
