import numpy as np


class Normalization:
    def __init__(self, X):
        self.min_vals = []
        self.intervals = []
        self.normalize(X)

    def normalize_column(self, col) -> None:
        min_val = np.min(col)
        max_val = np.max(col)
        interval_len = max_val - min_val
        for i in range(len(col)):
            col[i] = (col[i] - min_val) / interval_len
        self.min_vals.append(min_val)
        self.intervals.append(interval_len)

    def normalize(self, X) -> None:
        col_count = X.shape[1]
        for j in range(col_count):
            self.normalize_column(X[:, j])

    def normalize_value(self, val: float, coord_ind: int) -> float:
        return (val - self.min_vals[coord_ind]) / self.intervals[coord_ind]

    def normalize_point(self, point: [float]) -> None:
        for i in range(len(point)):
            point[i] = self.normalize_value(point[i], i)
