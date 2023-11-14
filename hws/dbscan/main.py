import pygame
import random
import numpy as np
from algorithm import give_flags, clusterize
from utils import dist


POINT_RADIUS = 3


def generate_random_points(coord: (float, float)) -> [(float, float)]:
    count = random.randint(2, 5)
    points = []
    for i in range(count):
        angle = np.pi * random.randint(0, 360) / 180
        radius = random.randint(10, 20)
        x = radius * np.cos(angle) + coord[0]
        y = radius * np.sin(angle) + coord[1]
        points.append((x, y))
    return points


def run():
    pygame.init()
    screen = pygame.display.set_mode((600, 400), pygame.RESIZABLE)
    screen.fill(color='white')
    pygame.display.update()

    points = []
    greens = []
    yellows = []
    reds = []
    neighbours = []
    is_pressed = False
    running = True
    dbscan_state = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE:
                screen.fill(color='white')
                for point in points:
                    pygame.draw.circle(screen, color='black', center=point, radius=POINT_RADIUS)

            elif event.type == pygame.KEYUP:
                if event.key == 13:  # Enter
                    screen.fill(color='white')
                    points = []
                elif event.key == 9:  # Tab
                    if dbscan_state == 0:
                        dbscan_state = 1
                        greens, yellows, reds, neighbours = give_flags(points)
                        for point_idx in greens:
                            pygame.draw.circle(screen, color='green', center=points[point_idx],
                                               radius=POINT_RADIUS)
                        for point_idx in yellows:
                            pygame.draw.circle(screen, color='yellow', center=points[point_idx],
                                               radius=POINT_RADIUS)
                        for point_idx in reds:
                            pygame.draw.circle(screen, color='red', center=points[point_idx],
                                               radius=POINT_RADIUS)
                    elif dbscan_state == 1:
                        screen.fill(color='white')
                        dbscan_state = 0
                        clusters = clusterize(greens, yellows, neighbours, points)
                        cluster_count = len(clusters)
                        print('number of clusters -', cluster_count)
                        cluster_color = [
                            [random.randint(0, 255),
                             random.randint(0, 255),
                             random.randint(0, 255)] for _ in range(cluster_count)
                        ]
                        for i, cluster in enumerate(clusters):
                            for point_idx in cluster:
                                pygame.draw.circle(
                                    screen,
                                    color=cluster_color[i],
                                    center=points[point_idx],
                                    radius=POINT_RADIUS
                                )

                        for point_idx in reds:
                            pygame.draw.circle(screen, color='red', center=points[point_idx], radius=POINT_RADIUS)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    is_pressed = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    is_pressed = False

            if is_pressed:
                coord = event.pos
                if len(points) > 0:
                    if dist(points[-1], coord) > 5 * POINT_RADIUS:
                        pygame.draw.circle(screen, color='black', center=coord, radius=POINT_RADIUS)
                        near_points = generate_random_points(coord)
                        points += near_points
                        for elem in near_points:
                            pygame.draw.circle(screen, color='black', center=elem, radius=POINT_RADIUS)
                        points.append(coord)
                else:
                    points.append(coord)
                    pygame.draw.circle(screen, color='black', center=coord, radius=POINT_RADIUS)

            pygame.display.update()

    pygame.quit()


if __name__ == '__main__':
    run()
