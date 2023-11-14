import os

import numpy as np


K_FILENAME = 'value_of_k.txt'


def calc_dist(a, b) -> float:
    return np.sqrt(
        (a[0] - b[0]) ** 2
        + (a[1] - b[1]) ** 2
        + (a[2] - b[2]) ** 2
        + (a[3] - b[3]) ** 2
    )


def find_minimums(point, X, minimums_count: int) -> [int]:
    indices_of_min = [None] * minimums_count
    min_dists = [float('+inf')] * minimums_count

    for i in range(minimums_count):
        for j in range(len(X)):
            if j in indices_of_min:
                continue
            dist = calc_dist(point, X[j])
            if dist < min_dists[i]:

                min_dists[i] = dist
                indices_of_min[i] = j

    return indices_of_min


def find_mode(lst: [int]) -> int:
    return max(set(lst), key=lst.count)


def predict_class(point, X, y, num_of_min: int):
    indices_of_min = find_minimums(point, X, num_of_min)
    possible_classes = [y[ind] for ind in indices_of_min]
    cls = find_mode(possible_classes)

    return cls


def calc_correctness(predicted: [int], labels: [int]) -> float:
    correct = 0
    n = len(predicted)
    for i in range(n):
        if predicted[i] == labels[i]:
            correct += 1
    return correct / n


def persist_k(k: int) -> None:
    with open(K_FILENAME, 'w') as f:
        f.write(str(k))


def retrieve_k_from_file() -> int:
    with open(K_FILENAME, 'r') as f:
        k = int(f.read())
    return k


def check_k_file() -> bool:
    return os.path.exists(K_FILENAME)


def define_k(train, test):
    tries = [0]
    for k in range(1, len(train[0])):
        predicted_classes = []
        for point in test[0]:
            predicted_classes.append(predict_class(point, train[0], train[1], k))
        correctness = calc_correctness(predicted_classes, test[1])
        print(k, correctness)
        tries.append(correctness)

    max_correctness = tries[0]
    k = 0
    for i in range(1, len(tries)):
        if tries[i] > max_correctness:
            max_correctness = tries[i]
            k = i

    persist_k(k)

    return k


def analyze_new_points(X, y, k: int, normalizing) -> None:
    while True:
        print('Enter 4 values separated by spaces')
        try:
            point = [float(val) for val in input().split()]
        except ValueError:
            print('Incorrect values')
            continue
        if len(point) != 4:
            print('Enter exactly 4 values')
            continue
        normalizing.normalize_point(point)
        print('Normalized:', *point)
        cls = predict_class(point, X, y, k)
        print('Class -', cls)
        print('\n')
