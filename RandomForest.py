from sklearn.ensemble import RandomForestRegressor
import numpy as np


def random_forest(X, Y):
    size = len(X)
    idx = range(size)
    idx = np.array(idx)
    # shuffle the data
    np.random.shuffle(idx)
    rf = RandomForestRegressor(n_estimators=1000, min_samples_leaf=1)
    rf.fit(X[idx], Y[idx])
    return rf