"""
Obstacle Map for Path Planning
"""
import geometry


class ObstacleMap:
    """
    Class for obstacle map, to be used for A* algorithm.

    Attributes
    ----------
    polygons: list of geometry.Polygon
        List of polygons, in the form [[y0, x0], [y1, x1], [y2, x2], ..., [yn, xn]].
    ellipses: list of np.array
        List of ellipses, in the form [center_y, center_x, h, w].
    """

    def __init__(self, height, width, distance, polygons=None, ellipses=None):
        """
        Initialization of the obstacle map.

        Parameters
        ----------
        distance: float
            Minimum distance between the center of the robot from any point in the obstacle space.
            Accounts for both minimum clearance and the radius of the robot.
        polygons: list of geometry.Polygon
            List of sets of point coordinates, with each set of coordinates forming a polygonal obstacle, to add to the
            obstacle map.
        ellipses: list of list
            List of sets of ellipse attributes, with each set of attributes forming an elliptical obstacle, to add to
            the obstacle map.
        """
        self.height = height
        self.width = width
        self.thickness = distance
        self.polygons = polygons if polygons is not None else []
        self.ellipses = ellipses if ellipses is not None else []

    def is_colliding(self, point, thickness=None, check_inside=True):
        """
        This function calculates the relative angle between the point and all the vertices of the image (between
        -180 and 180 degrees). If the sign of this direction ever changes, the point lies outside of the polygon.
        Otherwise, the point must lie outside of the polygon.

        Parameters
        ----------
        point: iterable
            The point to check for collisions, in the form (y, x).
        thickness: float
            Distance to use for rigid robot.  Default uses radius and clearance.
        check_inside: bool
            Whether to check inside the polygon.

        Returns
        -------
        bool
            True if there is a collision, False otherwise.
        """
        t = self.thickness if thickness is None else thickness
        ry = point[0]
        rx = point[1]
        if not (t <= ry <= self.height - t) or not (t <= rx <= self.width - t):
            return True

        # Ellipses / circles
        for ellipse in self.ellipses:
            vy, vx, vh, vw = ellipse
            vw += t
            vh += t
            if ((vx - rx) ** 2) + (((vy - ry) * vw / vh) ** 2) <= vw ** 2.0:
                return True

        # Polygons
        for polygon in self.polygons:
            if polygon.is_colliding(ry, rx, t, check_inside=check_inside):
                return True
        return False
