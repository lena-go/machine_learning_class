import math

import matplotlib.pyplot as plt
import numpy as np


class Drawer:
    def __init__(self, clf):
        self.cur_fig = 0
        self.clf = clf
        self.w = np.hstack([self.clf.coef_[0], clf.intercept_[0]])  # vector orthogonal to the hyperplane

    def draw_sup(self, xx) -> None:
        c = self.clf.support_vectors_[0]
        a = - self.w[0] / self.w[1]
        yy_down = a * xx + (c[1] - a * c[0])
        c = self.clf.support_vectors_[-1]
        yy_up = a * xx + (c[1] - a * c[0])

        plt.plot(xx, yy_down, 'k--', alpha=0.6)
        plt.plot(xx, yy_up, 'k--', alpha=0.6)

        plt.scatter(
            self.clf.support_vectors_[:, 0],
            self.clf.support_vectors_[:, 1],
            s=100,
            linewidths=1,
            facecolors='none',
            edgecolors="k"
        )

    def draw(self, X, y) -> None:
        plt.figure(self.cur_fig)
        self.cur_fig += 1

        if self.w[1] == 0:
            num_points = 50
            x = - self.w[2] / self.w[0]
            xx = np.array([x] * num_points)
            min_y = np.min(X[:, 1])
            max_y = np.max(X[:, 1])
            yy = np.linspace(math.floor(min_y), math.ceil(max_y), num_points)
        else:
            min_x = np.min(X[:, 0])
            max_x = np.max(X[:, 0])
            xx = np.linspace(math.floor(min_x), math.ceil(max_x))
            a = - self.w[0] / self.w[1]
            b = - self.w[2] / self.w[1]
            yy = a * xx + b

        plt.plot(xx, yy, 'k-', alpha=0.6)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Set3)

        plt.axis('tight')

    @staticmethod
    def show():
        plt.show()
