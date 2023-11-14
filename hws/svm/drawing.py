import math

import matplotlib.pyplot as plt
import numpy as np


class Drawer:
    def __init__(self, clf):
        self.cur_fig = 0
        self.clf = clf
        w = self.clf.coef_[0]  # vector orthogonal to the hyperplane
        self.a = -w[0] / w[1]  # y = ax + b
        self.b = - (clf.intercept_[0]) / w[1]

    def draw_sup(self, xx) -> None:
        c = self.clf.support_vectors_[0]
        yy_down = self.a * xx + (c[1] - self.a * c[0])
        c = self.clf.support_vectors_[-1]
        yy_up = self.a * xx + (c[1] - self.a * c[0])

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

    def draw(self, X, y, display_sup: bool = False) -> None:
        plt.figure(self.cur_fig)
        self.cur_fig += 1
        min_x = np.min(X[:, 0])
        max_x = np.max(X[:, 0])
        xx = np.linspace(math.floor(min_x), math.ceil(max_x))

        yy = self.a * xx + self.b
        plt.plot(xx, yy, 'k-', alpha=0.6)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Set3)

        if display_sup:
            self.draw_sup(xx)

        plt.axis('tight')

    @staticmethod
    def show():
        plt.show()
