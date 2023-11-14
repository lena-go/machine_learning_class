import random
from itertools import combinations

from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

from normalization import Normalization
from algorithm import (
    define_k,
    retrieve_k_from_file,
    check_k_file,
    analyze_new_points
)


def shuffle_samples(X, y):
    zipped = list(zip(X, y))
    random.Random(0).shuffle(zipped)
    shuffled_X, shuffled_y = zip(*zipped)
    return np.array(shuffled_X), np.array(shuffled_y)


def divide_on_train_and_test(X, y, part: float = 0.8):
    shuffled_X, shuffled_y = shuffle_samples(X, y)
    elements_count = int(part * len(shuffled_y))
    return ((shuffled_X[:elements_count], shuffled_y[:elements_count]),
            (shuffled_X[elements_count:], shuffled_y[elements_count:]))


def projection(X, y):
    fig, axs = plt.subplots(2, 3)
    comb = list(combinations((0, 1, 2, 3), 2))
    col_names = ["sepal length", "sepal width", "petal length", "petal width"]

    for i in range(2):
        for j in range(3):
            axs[i][j].scatter(
                X[:,  comb[i * 3 + j][0]],
                X[:,  comb[i * 3 + j][1]],
                c=y
            )
            axs[i][j].set_title(f"Projection on {col_names[comb[i * 3 + j][0]]} and {col_names[comb[i * 3 + j][1]]}")
            axs[i][j].set_xlabel(col_names[comb[i * 3 + j][0]])
            axs[i][j].set_ylabel(col_names[comb[i * 3 + j][1]])


def run():
    X = load_iris()['data']
    y = load_iris()['target']
    projection(X, y)
    normalizing = Normalization(X)
    projection(X, y)
    train, test = divide_on_train_and_test(X, y)

    if check_k_file():
        k = retrieve_k_from_file()
    else:
        k = define_k(train, test)

    plt.show()

    analyze_new_points(train[0], train[1], k, normalizing)


if __name__ == '__main__':
    run()
