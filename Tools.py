from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt


def load_boston_data():
    boston = load_boston()
    X = boston["data"]
    Y = boston["target"]
    print(X)
    return X, Y


def show_data(*data):
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, v in enumerate(data):
        plt.plot(v, color=color[i])
    plt.show()


def pred_ints(model, X, percentile=95, t=1):
    err_down = []
    err_up = []
    for x in range(len(X)):
        preds = []
        for pred in model.estimators_:
            preds.append(pred.predict([X[x]]))
        err_down.append(np.percentile(preds, (100 - percentile) / 2. ))
        err_up.append(np.percentile(preds, 100 - (100 - percentile) / 2.))
    err_down = np.array(err_down)
    err_up = np.array(err_up)
    err_down *= t
    err_up *= t
    return err_down, err_up

