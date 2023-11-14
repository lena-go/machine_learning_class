import matplotlib.pyplot as plt
import matplotlib.colors as m_colors
import random
import numpy as np


class Point:
    def __init__(self, x, y, cluster=0) -> None:
        self.x = x
        self.y = y
        self.cluster = cluster
        self.prev_cluster = -1


class Centroid(Point):
    def __init__(self, x, y, cluster=0) -> None:
        super().__init__(x, y, cluster)
        self.prev_x = None
        self.prev_y = None


def make_random_points(n: int = 100) -> [Point]:
    points = []
    for i in range(n):
        points.append(Point(
            random.randrange(0, 100),
            random.randrange(0, 100)
        ))

    plt.figure(0)
    for point in points:
        plt.scatter(point.x, point.y, color='pink')

    return points


def calc_dist(point_a: Point, point_b: Point) -> float:
    return np.sqrt(
        (point_a.x - point_b.x) ** 2
        + (point_a.y - point_b.y) ** 2
    )


def make_first_centroids(points: [Point], k: int = 4, to_plot: bool = True) -> [Point]:
    circle_center = Point(0, 0)

    for point in points:
        circle_center.x += point.x
        circle_center.y += point.y

    circle_center.x /= len(points)
    circle_center.y /= len(points)

    r = 0
    for point in points:
        d = calc_dist(point, circle_center)
        if d > r:
            r = d

    centroids = []
    for j in range(k):
        centroids.append(Centroid(
            round(r * np.cos(2 * np.pi * j / k) + circle_center.x, 1),
            round(r * np.sin(2 * np.pi * j / k) + circle_center.y, 1),
            cluster=j
        ))

    if to_plot:
        plt.figure(0)
        for centroid in centroids:
            plt.scatter(
                centroid.x,
                centroid.y,
                c='black',
                marker='+',
            )

    return centroids


def get_colors(num_centroids: int = 4) -> [str]:
    # BASE_COLORS
    # CSS4_COLORS
    # TABLEAU_COLORS
    # XKCD_COLORS
    return random.sample(list(m_colors.XKCD_COLORS.keys()), k=num_centroids)


def draw(points: [Point], centroids: [Centroid], colors: [str], figure_num: int) -> None:
    plt.figure(figure_num)
    for point in points:
        plt.scatter(point.x, point.y, c=colors[point.cluster])

    for centroid in centroids:
        plt.scatter(
            centroid.x,
            centroid.y,
            c='black',
            marker='+',
        )


def recalc_centroids(points: [Point], centroids: [Centroid]) -> None:
    num_points_in_cluster = [0 for _ in range(len(centroids))]

    for centroid in centroids:
        centroid.prev_x = centroid.x
        centroid.prev_y = centroid.y
        centroid.x = 0
        centroid.y = 0

    for point in points:
        centroids[point.cluster].x += point.x
        centroids[point.cluster].y += point.y
        num_points_in_cluster[point.cluster] += 1

    for j in range(len(centroids)):
        if num_points_in_cluster[j] != 0:
            centroids[j].x = round(centroids[j].x / num_points_in_cluster[j], 1)
            centroids[j].y = round(centroids[j].y / num_points_in_cluster[j], 1)
        else:
            centroids[j].x = centroids[j].prev_x
            centroids[j].y = centroids[j].prev_y


def time_to_finish(points: [Point], centroids: [Centroid]) -> bool:
    were_clusters_changed = False
    for point in points:
        if point.prev_cluster != point.cluster:
            were_clusters_changed = True
            break

    were_centroids_changed = False
    for centroid in centroids:
        if (centroid.prev_x != centroid.x) or (centroid.prev_y != centroid.y):
            were_centroids_changed = True
            break

    return (not were_clusters_changed) or (not were_centroids_changed)


def do_step(points: [Point], centroids: [Centroid], colors: [str], step_num: int, to_plot: bool = True) -> None:

    num_figures = 2

    for point in points:
        dists_to_centroids = [calc_dist(point, centroid) for centroid in centroids]
        point.prev_cluster = point.cluster
        point.cluster = np.argmin(dists_to_centroids)

    if to_plot:
        draw(points, centroids, colors, step_num * num_figures + 2)

    recalc_centroids(points, centroids)

    if to_plot:
        draw(points, centroids, colors, step_num * num_figures + 3)


def do_steps(points: [Point], centroids: [Centroid], colors: [str], to_plot: bool = True) -> None:
    step = 0
    do_step(points, centroids, colors, step, to_plot)
    while not time_to_finish(points, centroids):
        step += 1
        do_step(points, centroids, colors, step, to_plot)


def calc_num_clusters(points: [Point]) -> int:
    dist_to_clusters_on_k = [float('inf')]

    for j in range(1, int(0.2 * len(points) + 1)):
        dist_to_clusters = [0 for _ in range(j)]
        colors = ['black' for _ in range(j)]
        centroids = make_first_centroids(points, k=j, to_plot=False)
        do_steps(points, centroids, colors, to_plot=False)

        for point in points:
            dist = calc_dist(point, centroids[point.cluster])
            dist_to_clusters[point.cluster] += (dist * dist)

        dist_to_clusters_on_k.append(sum(dist_to_clusters))

    plt.figure(1)
    plt.plot(
        range(1, int(0.2 * len(points) + 1)),
        dist_to_clusters_on_k[1:]
    )

    D = []
    print('check')
    for j in range(2, len(dist_to_clusters_on_k) - 1):
        print('j', j)
        print(dist_to_clusters_on_k[j])
        if np.abs(dist_to_clusters_on_k[j - 1] - dist_to_clusters_on_k[j]) != 0:
            D.append(
                np.abs(dist_to_clusters_on_k[j] - dist_to_clusters_on_k[j + 1])
                / np.abs(dist_to_clusters_on_k[j - 1] - dist_to_clusters_on_k[j])
            )
        else:
            D.append(float('inf'))

    print(D)
    num_clusters = np.argmin(D).item() + 2
    print('number of clusters is', num_clusters)

    return num_clusters


def run():
    points = make_random_points()
    num_clusters = calc_num_clusters(points)
    colors = get_colors(num_clusters)
    centroids = make_first_centroids(points, num_clusters)
    do_steps(points, centroids, colors)
    plt.show()


if __name__ == '__main__':
    run()
