import random

from sklearn import svm
from sklearn.datasets import make_blobs
import numpy as np

from drawing import Drawer


def randomize_points(n: int, mode: int = 0):
    half = n // 2
    if mode == 0:
        return make_blobs(n_samples=n, centers=2)  # random_state=6
    if mode == 1:
        # np.random.seed(0)
        X = np.r_[
            np.random.randn(half, 2) - [2, 2],
            np.random.randn(half, 2) + [2, 2]
        ]
        print(X)
        y = [0]*half + [1]*half
        return X, y


def add_new_point(clf, X, y, point: [int, int] = None):
    if not point:
        point = [random.randint(-5, 5), random.randint(-5, 5)]
    predicted_cl = clf.predict([point])
    print('new point -', point)
    print('of class ', predicted_cl[0])
    new_X = np.append(X, [point], axis=0)
    new_y = np.append(y, predicted_cl)
    return new_X, new_y


def run():
    X, y = randomize_points(50, mode=0)
    clf = svm.SVC(kernel="linear", C=1000)
    clf.fit(X, y)
    my_plot = Drawer(clf)
    my_plot.draw(X, y, True)
    new_X, new_y = add_new_point(clf, X, y)
    my_plot.draw(new_X, new_y, True)
    my_plot.show()


if __name__ == '__main__':
    run()
