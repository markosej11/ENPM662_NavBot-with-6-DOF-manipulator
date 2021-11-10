""""""
import numpy as np
from math import *


class Polygon:
    def __init__(self, points):
        self.points = points if issubclass(type(points), np.ndarray) else np.array(points, dtype=np.float64)
        min_y = np.min(self.points[:, 0])
        min_x = np.min(self.points[:, 1])
        max_y = np.max(self.points[:, 0])
        max_x = np.max(self.points[:, 1])
        self.y = (min_y + max_y) / 2.0
        self.x = (min_x + max_x) / 2.0
        self.h = max_y - min_y
        self.w = max_x - min_x
        self.subpolygons = [self.points]

    def is_colliding(self, ry, rx, t, check_inside=True):
        # Skip the complex collision check if point of interest is not within bounds.
        # TODO: Figure out why this doesn't work...
        # if abs(ry - self.y) > (self.h/2 + t) or abs(rx - self.x) > (self.w/2 + t):
        #     return False

        # Complex collision check
        for polygon in self.subpolygons:
            direction = 0.0
            collision = True
            for j in range(len(polygon) + 1):
                # Check if the point is within range of any of the vertices
                py, px = polygon[j % len(polygon)]
                if ((ry - py) ** 2 + (rx - px) ** 2) <= (t ** 2):
                    return True
                # Check if the point is inside of the polygon
                elif not check_inside:
                    collision = False
                    continue
                vy, vx = polygon[(j + 1) % len(polygon)]
                new_direction = atan2(vy - ry, vx - rx)
                new_direction = new_direction if new_direction >= 0.0 else (new_direction + 2.0 * pi)
                difference = new_direction - direction
                difference = difference if difference >= -pi else difference + 2.0 * pi
                difference = difference if difference <= pi else difference - 2.0 * pi
                direction = new_direction
                if j > 0 > difference:
                    collision = False
            if collision:
                return True
            else:
                # Check if the point is within a certain distance of an edge
                for j in range(len(polygon)):
                    y0, x0 = (ry, rx)
                    y1, x1 = polygon[j]
                    y2, x2 = polygon[(j + 1) % len(polygon)]
                    d = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / sqrt(
                        (y2 - y1) ** 2 + (x2 - x1) ** 2)
                    if d < t:
                        dot_product = (x0 - x1) * (x0 - x2) + (y0 - y1) * (y0 - y2)
                        if dot_product < 0:
                            return True
        return False
