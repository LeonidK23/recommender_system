import random
import numpy as np

def cross_validation(recommender, data, k=10):
    all_data = np.array(data)
    all_indices = [ind for ind in range(len(data))]
    batch_size = len(data)//(k-1)
    rows_to_throw = []
    s = 0
    for i in range(k):
        if i < k-1:
            rows_to_throw.append(np.random.choice(all_indices, size=batch_size, replace=False).tolist())
        else:
            rows_to_throw.append(all_indices)
        all_indices = list(set(all_indices) - set(rows_to_throw[i]))

    all_indices = [ind for ind in range(len(data))]
    for rows in rows_to_throw:
        holdout_train = all_data[np.ix_(list(set(all_indices) - set(rows)))]
        holdout_test = all_data[np.ix_(rows)]
        
