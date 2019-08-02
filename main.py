import os
import numpy as np

from recommenders.mean_recommender import MeanRecommender
from recommenders.random_recommender import RandomRecommender
from recommenders.knn_recommender import kNNRecommender
from auxiliary.cross_validation import cross_validation

X_test = np.genfromtxt(os.path.join("data", "qualifying_blanc.csv"), delimiter=",", dtype=np.int)
X_train = np.genfromtxt(os.path.join("data", "train.csv"), delimiter=",", dtype=np.int)

clf_mean = MeanRecommender()
clf_mean.fit(X_train)
predictions_mean = clf_mean.predict(X_test)

clf_rand = RandomRecommender()
clf_rand.fit(X_train)
predictions_rand = clf_rand.predict(X_test)

clf_knn = kNNRecommender(9)
clf_knn.fit(X_train)
predictions_kNN = clf_knn.predict(X_test)

# clf = RandomRecommender()
# clf = MeanRecommender()
# clf = kNNRecommender(9)
# acc = cross_validation(clf, X_train)

np.savetxt("data/qualifying_mean.csv", predictions_mean,
           delimiter=",", newline="\n", encoding="utf-8")
np.savetxt("data/qualifying_random.csv", predictions_rand,
           delimiter=",", newline="\n", encoding="utf-8")
np.savetxt("data/qualifying_knn.csv", predictions_kNN,
          delimiter=",", newline="\n", encoding="utf-8")
