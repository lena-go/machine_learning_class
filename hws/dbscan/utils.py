import numpy as np


def dist(point_a: (float, float), point_b: (float, float)) -> float:
    return np.sqrt(((point_a[0] - point_b[0]) ** 2) + (point_a[1] - point_b[1]) ** 2)
