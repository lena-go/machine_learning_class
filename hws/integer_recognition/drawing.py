from math import floor

import pygame
import numpy as np
from sklearn.cluster import DBSCAN
from PIL import Image, ImageDraw, ImageOps


CIRCLE_RADIUS = 3
ENTER_KEY = 13
TAB_KEY = 9
BG_COLOR = 'black'
BRUSH_COLOR = 'white'
SCREENSIZE = (600, 600)
PADDING_RATIO = 5 / 28
BRUSH_SIZE_RATIO = 2 / 28


def dist(point_a: (float, float), point_b: (float, float)) -> float:
    return np.sqrt(((point_a[0] - point_b[0]) ** 2) + (point_a[1] - point_b[1]) ** 2)


class Desk:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(SCREENSIZE)
        self.screen.fill(color=BG_COLOR)
        pygame.display.update()
        self.points = []
        self.labels = []
        self.num_classes = 0
        self.run()

    def run(self):
        running = True
        is_pressed = False
        self.points = []
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYUP:
                    if event.key == TAB_KEY:
                        self.screen.fill(color=BG_COLOR)
                        self.points = []
                    elif event.key == ENTER_KEY and self.points:
                        self.clusterize()
                        self.make_pictures()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        is_pressed = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        is_pressed = False

                if is_pressed:
                    coord = list(event.pos)
                    pygame.draw.circle(
                        self.screen,
                        color=BRUSH_COLOR,
                        center=coord,
                        radius=CIRCLE_RADIUS
                    )
                    self.points.append(coord)

                pygame.display.update()

    def clusterize(self) -> None:
        clustering = DBSCAN(eps=10, min_samples=4).fit(self.points)
        self.labels = clustering.labels_

    def make_pictures(self):
        classes = set(self.labels)
        self.num_classes = len(classes)
        if -1 in classes:
            self.num_classes -= 1
        coordinates_by_class = [[] for _ in range(self.num_classes)]

        for i, label in enumerate(self.labels):
            if label == -1:
                continue
            coordinates_by_class[label].append(self.points[i])

        for cls in coordinates_by_class:
            self.make_picture(cls)

    @staticmethod
    def make_picture(coordinates: [(int, int)], show: bool = False):
        coords_np = np.asarray(coordinates)
        mean = np.mean(coords_np, axis=0)
        delta = np.array((SCREENSIZE[0] / 2, SCREENSIZE[1] / 2)) - mean
        shifted_coords = coords_np + delta
        img = Image.new('RGB', SCREENSIZE, BG_COLOR)
        draw = ImageDraw.Draw(img)

        xx = shifted_coords[:, 0]
        yy = shifted_coords[:, 1]
        min_x = xx.min()
        min_y = yy.min()
        max_x = xx.max()
        max_y = yy.max()

        shifted_mean = np.mean(shifted_coords, axis=0)
        half_square_size = max(
            abs(shifted_mean[0] - min_x),
            abs(shifted_mean[1] - min_y),
            abs(shifted_mean[0] - max_x),
            abs(shifted_mean[1] - max_y)
        )

        padding = int(PADDING_RATIO * half_square_size * 2)

        # crop right by frame
        # to_crop = (
        #     floor(min_x) - padding,
        #     floor(min_y) - padding,
        #     floor(SCREENSIZE[0] - max_x) - padding,
        #     floor(SCREENSIZE[1] - max_y) - padding
        # )

        to_crop = (
            floor(shifted_mean[0] - half_square_size) - padding,
            floor(shifted_mean[1] - half_square_size) - padding,
            floor(SCREENSIZE[0] - shifted_mean[0] - half_square_size) - padding,
            floor(SCREENSIZE[1] - shifted_mean[1] - half_square_size) - padding,
        )

        brush_size = int(BRUSH_SIZE_RATIO * half_square_size * 2)
        for (x, y) in shifted_coords:
            draw.rectangle(
                [x, y, x + brush_size, y + brush_size],
                fill=BRUSH_COLOR,
            )
        cropped = ImageOps.crop(img, to_crop)
        if show:
            img.show()
            cropped.show()
        cropped.thumbnail((28, 28))
        if show:
            cropped.show()


def run():
    desk = Desk()
    desk.run()


if __name__ == '__main__':
    run()
