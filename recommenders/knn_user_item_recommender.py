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

    # def fit(self, train_data):
    #     self.global_mean = np.mean(train_data[:, 2])
    #     sparse_mat = csr_matrix((train_data[:, 2], (train_data[:, 0], train_data[:, 1])))
    #     self.train_data = sparse_mat
    #     simil_matrix_users = cosine_similarity(sparse_mat)
    #     np.fill_diagonal(simil_matrix_users, -1)
    #     k_nearest_users_dists = np.sort(simil_matrix_users, axis=1)[:, -self.n_neighbours:]
    #     self.k_nearest_users = np.array([self.get_neighbour_indices(simil_matrix_users[i, :], k_nearest_users_dists[i, :])
    #                           for i in range(k_nearest_users_dists.shape[0])])
    #     # Find nearest neighbours for each item
    #     simil_matrix_items = cosine_similarity(sparse_mat.transpose())
    #     np.fill_diagonal(simil_matrix_items, -1)
    #     k_nearest_items_dists = np.sort(simil_matrix_items, axis=1)[:, -self.n_neighbours:]
    #     self.k_nearest_items = np.array([self.get_neighbour_indices(simil_matrix_items[i, :], k_nearest_items_dists[i, :])
    #                   for i in range(k_nearest_items_dists.shape[0])])
    #
    def get_neighbour_indices(self, user, distances):
        inds = np.array([np.where(user == dist)[0][0] for dist in distances])

        return inds

    def collect_user_items_dict(self, test_data):
        items_to_fill = {}
        for entry in test_data:
            if entry[0] not in items_to_fill:
                items_to_fill[entry[0]] = []
            items_to_fill[entry[0]] += [entry[1].item()]

        return items_to_fill

    def collect_items_user_dict(self, test_data):
        items_to_fill = {}
        for entry in test_data:
            if entry[1] not in items_to_fill:
                items_to_fill[entry[1]] = []
            items_to_fill[entry[1]] += [entry[0].item()]

        return items_to_fill

    # def predict(self, test_data):
    #     predictions = []
    #     users_items_to_fill = self.collect_user_items_dict(test_data)
    #     for user in users_items_to_fill:
    #         for item in users_items_to_fill[user]:
    #             neighbours_items = self.train_data[np.ix_([user], self.k_nearest_items[item])].A
    #             neighbours_users = self.train_data[np.ix_(self.k_nearest_users[user], [item])].A
    #             other_neighbours = self.train_data[np.ix_(self.k_nearest_users[user], self.k_nearest_items[item])].A
    #
    #             nearest_items_prediction = np.mean(neighbours_items)
    #             nearest_users_prediction = np.mean(neighbours_users)
    #             prediction = nearest_users_prediction if nearest_items_prediction == 0 else nearest_items_prediction
    #             if prediction == 0:
    #                 prediction = np.mean(other_neighbours)
    #
    #             predictions.append([user, item, prediction if prediction != 0 else 0])
    #
    #     return np.array(predictions)

    def pearson_corr(self, mat):
        mu = np.zeros((mat.shape[0], 1))
        non_zeros_items_indices = []
        corr_mat = np.zeros((mat.shape[0], mat.shape[0]))
        corr_mat[:] = np.nan
        non_zeros_items_indices = np.array(list(set(mat.nonzero()[0])))
        non_zeros_items = mat[non_zeros_items_indices, :]
        non_zeros_mu = mu[non_zeros_items_indices]
        centered_items = non_zeros_items - non_zeros_mu
        for i in range(mat.shape[0]):
            mu[i] = np.nan if mat[i, :].data.shape[0]==0 else np.mean(mat[i, :].data)
        for i in range(mat.shape[0]):
            I_u = mat[i, :].nonzero()[1]
            if I_u.shape[0] > 0:
                # center_i = mat[i, :] - mu[i]
                center_i = mat[i, :]
                numerator = np.sum(np.multiply(center_i, centered_items), axis=1)
                denumerator = np.sqrt(np.sum(np.square(center_i)))*np.sqrt(np.sum(np.square(centered_items), axis=1))
                corr_mat[i, non_zeros_items_indices] = np.divide(numerator, denumerator).reshape(-1)
            else:
                corr_mat[i, :] = np.nan

        return corr_mat

    def fit(self, train_data):
        # self.train_data = csr_matrix((train_data[:, 2], (train_data[:, 0], train_data[:, 1]))).todense().T
        # self.sim_mat_items = cosine_similarity(self.train_data)
        # np.fill_diagonal(self.sim_mat_items, -1)
        self.train_data = csr_matrix((train_data[:, 2], (train_data[:, 0], train_data[:, 1]))).T
        self.sim_mat_items = self.pearson_corr(self.train_data)

    def predict(self, test_data):
        # The best config
        # items_user = self.collect_items_user_dict(test_data)
        # predictions = []
        # for item in items_user:
        #     closest_dists = np.sort(self.sim_mat_items[item, :])[-self.n_neighbours:]
        #     nearest_items = []
        #     for dist in closest_dists:
        #         nearest_items.append(np.where(self.sim_mat_items[item, :] == dist)[0][0])
        #     nearest_feedbacks = self.train_data[np.array(nearest_items), :]
        #     for user in items_user[item]:
        #         prediction = np.sqrt(np.sum(np.square(nearest_feedbacks[:, user])))/self.n_neighbours*2.5
        #         # if nearest_feedbacks[:, user].data.shape[0] == 0:
        #         #     prediction = 0
        #         # else:
        #         #     prediction = np.mean(nearest_feedbacks[:, user].data)
        #         predictions.append([user, item, prediction])


        items_user = self.collect_items_user_dict(test_data)
        predictions = []
        for item in items_user:
            dists = np.sort(self.sim_mat_items[item, :])
            num_dists = dists[~np.isnan(dists)]
            nearest_dists = num_dists[-self.n_neighbours:]
            nearest_items_inds = []
            for dist in nearest_dists:
                nearest_items_inds.append(np.where(self.sim_mat_items[item, :] == dist)[0][0])
            nearest_items = self.train_data[np.array(nearest_items_inds), :]
            for user in items_user[item]:
                nearest_ratings = nearest_items[:, user].data
                ratings_inds = nearest_items[:, user].nonzero()[0]
                # print(nearest_dists)
                # print(nearest_ratings - np.mean(nearest_ratings))
                # print(nearest_ratings)
                # print(ratings_inds)
                # print(nearest_dists)
                # print(nearest_dists[ratings_inds])
                # print(np.sum(nearest_dists[ratings_inds]))
                # print('----------------------------------------')
                prediction = np.sum(np.multiply(nearest_dists[ratings_inds], nearest_ratings))/np.sum(nearest_dists[ratings_inds])
                predictions.append([user, item, np.mean(self.train_data[item, :]) if np.isnan(prediction) else prediction])

        return np.array(predictions)
