import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

class kNNRecommender:
    def __init__(self, k):
        self.n_neighbours = k
        self.train_data = 0

    def fit(self, train_data):
        self.train_data = csr_matrix((train_data[:, 2] + 1, (train_data[:, 0], train_data[:, 1]))).T
        # self.sim_mat_items = self.compute_pearson(train_data, beta=200)
        self.sim_mat_items = self.load_pearson()
        np.fill_diagonal(self.sim_mat_items, -1)

    def load_pearson(self):
        return np.load("data/pears_corr.npy")

    def compute_pearson(self, mat, beta=False):
        mat = csr_matrix((mat[:, 2] + 1, (mat[:, 0], mat[:, 1]))).T
        mu = np.zeros((mat.shape[0], 1))
        corr_mat = np.zeros((mat.shape[0], mat.shape[0]))
        corr_mat[:] = np.nan
        non_zeros = np.argwhere(mat > 0)                                        # known values indices
        I_v = []                                                                # each item of I_v is a list of users, which rated the item number I_v.index(item)
        for i in range(mat.shape[0]):
            # if there are no ratings mu = nan, otherwise mean value
            mu[i] = np.nan if mat[i, :].data.shape[0]==0 else np.mean(mat[i, :].data - 1)
            I_v.append(non_zeros[non_zeros[:, 0]==i][:, 1])

        for i in range(mat.shape[0]):
            I_u = I_v[i]
            if np.isnan(mu[i]) == False:                                        # if the item was rated by somebody
                for j in range(len(I_v)):
                    I_u_I_v = np.intersect1d(I_u, I_v[j])                       # get similar users
                    if I_u_I_v.shape[0] > 0:
                        centered_ratings_u = mat[i, I_u_I_v].data - 1 - mu[i]
                        centered_ratings_v = mat[j, I_u_I_v].data - 1 - mu[j]
                        numerator = np.sum(np.multiply(centered_ratings_u, centered_ratings_v))
                        denominator = np.sqrt(np.sum(np.square(centered_ratings_u)))* \
                                      np.sqrt(np.sum(np.square(centered_ratings_u)))
                        if denominator == 0:
                            denominator = 1e-6
                        if beta == False:
                            corr_mat[i, j] = numerator/denominator
                        else:
                            # significance weighting
                            corr_mat[i, j] = (I_u_I_v.shape[0]/beta)*numerator/denominator
                    else:
                        corr_mat[i, j] = np.nan
            else:
                corr_mat[i, :] = np.nan

        timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
        np.save('data/pears_corr_'+timestr+'.npy', corr_mat)

        return corr_mat

    def collect_items_user_dict(self, test_data):
        """Create a dictionary of items to predict and corresponding users."""
        items_to_fill = {}
        for row in test_data:
            user = row[0]
            item = row[1]
            if item not in items_to_fill:
                items_to_fill[item] = []
            items_to_fill[item] += [user.item()]

        return items_to_fill

    def predict(self, test_data):
        items_user = self.collect_items_user_dict(test_data)
        predictions = []
        for item in items_user:
            distances = np.sort(self.sim_mat_items[item, :])
            numerical_dists = distances[~np.isnan(distances)]
            nearest_dists = numerical_dists[-self.n_neighbours:]
            nearest_items_inds = []
            for dist in nearest_dists:
                nearest_items_inds.append(np.where(self.sim_mat_items[item, :] == dist)[0][0])
            nearest_items = self.train_data[np.array(nearest_items_inds), :]
            for user in items_user[item]:
                nearest_ratings = nearest_items[:, user].data - 1
                if nearest_ratings.shape[0] > 0:
                    nearest_ratings_inds = nearest_items[:, user].nonzero()[0]
                    prediction = np.sum(np.multiply(np.square(nearest_dists[nearest_ratings_inds]), \
                                 nearest_ratings))/np.sum(np.square(nearest_dists[nearest_ratings_inds]))
                else:
                    prediction = np.mean(self.train_data[item, :].data)
                predictions.append([user, item, prediction])

        return np.array(predictions)
