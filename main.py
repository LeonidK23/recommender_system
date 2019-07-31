import os
import numpy as np

from recommenders.mean_recommender import MeanRecommender
from recommenders.random_recommender import RandomRecommender
from recommenders.knn_user_user_recommender import kNNRecommenderUU
from recommenders.knn_user_item_recommender import kNNRecommenderUI
from recommenders.knn_item_item_recommender import kNNRecommenderII
from auxiliary.cross_validation import cross_validation

X_test = np.genfromtxt(os.path.join("data", "qualifying_blanc.csv"), delimiter=",", dtype=np.int)
X_train = np.genfromtxt(os.path.join("data", "train.csv"), delimiter=",", dtype=np.int)

# clf_mean = MeanRecommender()
# clf_mean.fit(X_train)
# predictions_mean = clf_mean.predict(X_test)
#
# clf_rand = RandomRecommender()
# clf_rand.fit(X_train)
# predictions_rand = clf_rand.predict(X_test)
#
# clf_knn_UU = kNNRecommenderUU(3)
# clf_knn_UU.fit(X_train)
# predictions_kNN_UU = clf_knn_UU.predict(X_test)
#
# clf_knn_II = kNNRecommenderII(3)
# clf_knn_II.fit(X_train)
# predictions_kNN_II = clf_knn_II.predict(X_test)
#
# clf_knn_UI = kNNRecommenderUI(9)
# clf_knn_UI.fit(X_train)
# predictions_kNN_UI = clf_knn_UI.predict(X_test)
#
# clf = RandomRecommender() # Error = 41.5
# clf = MeanRecommender() # Error = 5.5
# clf = kNNRecommenderUU(3) # Error = 4.3
# clf = kNNRecommenderII(3) # Error = appr. 3.7
clf = kNNRecommenderUI(9) # Error = 1.9
acc = cross_validation(clf, X_train)
# print(acc)

# np.savetxt("data/qualifying_mean.csv", predictions_mean,
#            delimiter=",", newline="\n", encoding="utf-8")
# np.savetxt("data/qualifying_random.csv", predictions_rand,
#            delimiter=",", newline="\n", encoding="utf-8")
# np.savetxt("data/qualifying_knn_UU.csv", predictions_kNN_UU,
#           delimiter=",", newline="\n", encoding="utf-8")
# np.savetxt("data/qualifying_knn_II.csv", predictions_kNN_II,
#           delimiter=",", newline="\n", encoding="utf-8")
# np.savetxt("data/qualifying_knn_UI.csv", predictions_kNN_UI,
#           delimiter=",", newline="\n", encoding="utf-8")
# np.savetxt("data/qualifying_hybrid.csv", final_pred,
#         delimiter=",", newline="\n", encoding="utf-8")

# TODO: add train_test_split
# TODO: cross_validation,
# TODO: correct packaging(add __init__ file for the project)
