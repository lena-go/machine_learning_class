import numpy as np
from utils import dist


def give_flags(
        points: [(float, float)],
        eps: int = 30,
        min_neighbours: int = 5
):
    if len(points) == 0:
        return []

    neighbours = [[] for _ in range(len(points))]
    greens = []
    yellows = []
    reds = []

    for i, point_a in enumerate(points):
        for j, point_b in enumerate(points):
            if i < j and dist(point_a, point_b) <= eps:
                neighbours[i].append(j)
                neighbours[j].append(i)

    for i, point_neighbours in enumerate(neighbours):
        if len(point_neighbours) >= min_neighbours:
            greens.append(i)

    for i, point_neighbours in enumerate(neighbours):
        if len(point_neighbours) < min_neighbours:
            has_green_neighbour = False
            for point_idx in point_neighbours:
                if point_idx in greens:
                    has_green_neighbour = True
                    yellows.append(i)
                    break

            if not has_green_neighbour:
                reds.append(i)

    return greens, yellows, reds, neighbours


def clusterize(
        greens: [int],
        yellows: [int],
        neighbours: [[int]],
        points: [(float, float)]
) -> ([[int]], [int]):
    green_clusters = clusterize_green(greens, neighbours)
    return clusterize_yellow(greens, yellows, neighbours, green_clusters, points)


def clusterize_green(
        greens: [int]
        , neighbours: [[int]]
) -> [[int]]:
    groups = []
    not_assigned = set(greens)
    while len(not_assigned) > 0:
        point = not_assigned.pop()
        groups.append([point])
        neighbours_to_check = neighbours[point]
        for neighbour in neighbours_to_check:
            if neighbour in greens and neighbour in not_assigned:
                groups[-1].append(neighbour)
                not_assigned.remove(neighbour)
                neighbours_to_check += neighbours[neighbour]

    return groups


def clusterize_yellow(
        greens: [int],
        yellows: [int],
        neighbours: [int],
        green_clusters: [[int]],
        points: [(float, float)]
):
    clusters = green_clusters[:]
    for point_idx in yellows:
        green_neighbours = []
        for neighbour_idx in neighbours[point_idx]:
            if neighbour_idx in greens:
                green_neighbours.append(neighbour_idx)

        if len(green_neighbours) == 1:
            for cluster in clusters:
                if green_neighbours[0] in cluster:
                    cluster.append(point_idx)
        else:
            distances = [
                dist(points[point_idx], points[neighbour_idx])
                for neighbour_idx in green_neighbours
            ]
            nearest_green_point = green_neighbours[np.argmin(distances)]
            for cluster in clusters:
                if nearest_green_point in cluster:
                    cluster.append(point_idx)

    return clusters
