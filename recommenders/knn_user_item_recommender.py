import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

class kNNRecommenderUI:
    def __init__(self, k):
        self.n_neighbours = k
        self.k_nearest_users = 0
        self.k_nearest_items = 0
        self.train_data = 0
        self.global_mean = 0
        self.current_I_u = 0

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

    def compute_similarity(self, other_items, I_u, mu_i, mu, I_v_list, I_v):
        I_u_I_v = np.intersect1d(I_u, I_v)
        if I_u_I_v.shape[0] > 0:
            ratings_u = other_items[i, I_u_I_v].data - 1 - mu[i]
            ratings_v = other_items[j, I_u_I_v].data - 1 - mu[I_v_list.index(I_v)]
            numerator = np.sum(np.multiply(ratings_u, ratings_v))
            denumerator = np.sqrt(np.sum(np.square(ratings_u)))*np.sqrt(np.sum(np.square(ratings_v)))
            return (i, j, numerator/denumerator)
            # print(numerator/denumerator)
            # print('-----------------')
        else:
            return (i, j, np.nan)

    def collect_I_u(self, I_v):
        return np.intersect1d(self.current_I_u, I_v)

    def pearson_corr(self, mat, beta=False):
        mat_ok = csr_matrix((mat[:, 2], (mat[:, 0], mat[:, 1]))).T
        mat_1 = csr_matrix((mat[:, 2]+1, (mat[:, 0], mat[:, 1]))).T
        mu = np.zeros((mat_ok.shape[0], 1))
        non_zeros_items_indices = []
        corr_mat = np.zeros((mat_ok.shape[0], mat_ok.shape[0]))
        corr_mat[:] = np.nan
        non_zeros = np.argwhere(mat_1 > 0)
        I_v = []
        for i in range(mat_ok.shape[0]):
            mu[i] = np.nan if mat_ok[i, :].data.shape[0]==0 else np.mean(mat_ok[i, :].data)
            I_v.append(non_zeros[non_zeros[:, 0]==i][:, 1])

        for i in range(mat_ok.shape[0]):
            I_u = mat_1[i, :].nonzero()[1]
            if np.isnan(mu[i]) == False:
                # func = partial(self.compute_similarity, mat_1, I_u, mu[i], mu)
                # with Pool(4) as p:
                #     print(p.starmap(self.compute_similarity, (mat_1, I_u, mu[i], mu, I_v)))
                for j in range(len(I_v)):
                    I_u_I_v = np.intersect1d(I_u, I_v[j])
                    if I_u_I_v.shape[0] > 0:
                        ratings_u = mat_1[i, I_u_I_v].data - 1 - mu[i]
                        ratings_v = mat_1[j, I_u_I_v].data - 1 - mu[j]
                        numerator = np.sum(np.multiply(ratings_u, ratings_v))
                        denumerator = np.sqrt(np.sum(np.square(ratings_u)))*np.sqrt(np.sum(np.square(ratings_v)))
                        if denumerator == 0:
                            denumerator = 1e-4
                        if beta == False:
                            corr_mat[i, j] = numerator/denumerator
                        else:
                            corr_mat[i, j] = numerator/denumerator*(min(I_u_I_v.shape[0], beta)/beta)
                        # print(ratings_u.shape)
                        # print(ratings_v.shape)
                        # print('-----------------')
                    else:
                        corr_mat[i, j] = np.nan
            else:
                corr_mat[i, :] = np.nan
            print(i)
        # corr_mat = np.load("data/pears_corr_20190729-141531.npy")
        # corr_mat = np.load("data/pears_corr_user_20190729-153839.npy")

            # if I_u.shape[0] > 0:
            #     # center_i = mat[i, :] - mu[i]
            #     center_i = mat[i, :]
            #     numerator = np.sum(np.multiply(center_i, centered_items), axis=1)
            #     denumerator = np.sqrt(np.sum(np.square(center_i)))*np.sqrt(np.sum(np.square(centered_items), axis=1))
            #     corr_mat[i, non_zeros_items_indices] = np.divide(numerator, denumerator).reshape(-1)
            # else:
            #     corr_mat[i, :] = np.nan

        import time
        timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
        np.save('data/pears_corr_'+timestr+'.npy', corr_mat)

        return corr_mat

    def fit(self, train_data):
        # self.train_data = csr_matrix((train_data[:, 2], (train_data[:, 0], train_data[:, 1]))).todense().T
        # self.sim_mat_items = cosine_similarity(self.train_data)
        # np.fill_diagonal(self.sim_mat_items, -1)
        self.train_data = csr_matrix((train_data[:, 2]+1, (train_data[:, 0], train_data[:, 1]))).T
        self.sim_mat_items = self.pearson_corr(train_data, beta=200)
        np.fill_diagonal(self.sim_mat_items, -1)

    def predict(self, test_data):
        # The best config
        # items_user = self.collect_items_user_dict(test_data)
        # predictions = []
        # for item in items_user:
        #     all_dists = np.sort(self.sim_mat_items[item, :])
        #     closest_dists = all_dists[~np.isnan(all_dists)][-self.n_neighbours:]
        #     nearest_items = []
        #     for dist in closest_dists:
        #         nearest_items.append(np.where(self.sim_mat_items[item, :] == dist)[0][0])
        #     nearest_feedbacks = self.train_data[np.array(nearest_items), :]
        #     print(nearest_feedbacks.shape)
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
                nearest_ratings = nearest_items[:, user].data - 1
                if nearest_ratings.shape[0] > 0:
                    ratings_inds = nearest_items[:, user].nonzero()[0]
                    # print(ratings_inds)
                    # print(nearest_ratings)
                    # print(nearest_dists[ratings_inds])
                    # print('----------------------------------------')
                    prediction = np.sum(np.multiply(nearest_dists[ratings_inds], nearest_ratings))/np.sum(nearest_dists[ratings_inds])
                else:
                    prediction = np.mean(self.train_data[:, user].data)
                    if np.isnan(prediction):
                        prediction = np.mean(self.train_data[item, :].data)
                    # prediction = 0

                predictions.append([user, item, np.mean(self.train_data[item, :]) if np.isnan(prediction) else prediction])




        # items_user = self.collect_user_items_dict(test_data)
        # predictions = []
        # for user in items_user:
        #     dists = np.sort(self.sim_mat_items[user, :])
        #     num_dists = dists[~np.isnan(dists)]
        #     nearest_dists = num_dists[-self.n_neighbours:]
        #     nearest_items_inds = []
        #     for dist in nearest_dists:
        #         nearest_items_inds.append(np.where(self.sim_mat_items[user, :] == dist)[0][0])
        #     nearest_items = self.train_data[np.array(nearest_items_inds), :]
        #     for item in items_user[user]:
        #         nearest_ratings = nearest_items[:, item].data - 1
        #         if nearest_ratings.shape[0] > 0:
        #             ratings_inds = nearest_items[:, item].nonzero()[0]
        #             # print(nearest_ratings)
        #             # print(ratings_inds)
        #             # print('----------------------------------------')
        #             numerator = np.sum(np.multiply(nearest_dists[ratings_inds], nearest_ratings))
        #             denumerator = np.sum(nearest_dists[ratings_inds])
        #             if denumerator == 0:
        #                 denumerator = 1e-4
        #             prediction = numerator/denumerator
        #         else:
        #             # prediction = np.mean(self.train_data[:, user].data)
        #             # if np.isnan(prediction):
        #             prediction = np.mean(self.train_data[user, :])
        #
        #         predictions.append([user, item, np.mean(self.train_data[user, :]) if np.isnan(prediction) else prediction])


        return np.array(predictions)
