from RandomForest import random_forest
from Tools import load_boston_data
import numpy as np
from Tools import show_data
from sklearn.model_selection import train_test_split
from Tools import pred_ints


def main():
    X, Y = load_boston_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state=2018)
    predictor = random_forest(X, Y)
    Y_predicted = predictor.predict(X_test)
    down, up = pred_ints(predictor, X_test, t=1)
    show_data(down, up)
    print(down[0])
    print(up[0])

if __name__ == '__main__':
    main()