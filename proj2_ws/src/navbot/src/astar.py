"""
A* Algorithm for Path Planning
"""
import numpy as np
from math import *
from obstacleMap import ObstacleMap
import logging


class AStar:
    def __init__(self, sy, sx, sr, gy, gx, rad=1.0, clr=0.1, d_theta=30, step=1.0, hw=1.0, cell_split=8, gt=-1,
                 obstacle_map=None):
        self.rad = rad
        self.clr = clr
        self.res = 0.5  # Resolution of matrix for tracking duplicate states, in units of meters
        self.theta = d_theta
        self.ti = 360 // self.theta
        self.map = ObstacleMap(10.0, 10.0, self.rad + self.clr) if obstacle_map is None else obstacle_map
        self.step = step  # Arc length of turn, per action
        self.cs = cell_split  # Arc length of turn, per action
        self.error = None

        # Handle argument values
        if self.rad < 0.0:
            self.error = ValueError("Radius is negative.  Exiting...")
            return
        elif self.clr < 0.0:
            self.error = ValueError("Clearance is negative.  Exiting...")
        else:
            if type(self.theta) is not int:
                self.error = ValueError("Use an integer for the turn angle coverage, in degrees.")
                return
            elif not (0 < self.theta <= 90) or 360 % self.theta != 0:
                self.error = ValueError("Make sure turn angle coverage is between 1 and 90 (inclusive), in degrees.")
                return

        # Set up actions (in terms of heading angles)
        self.actions = [self.theta * (i - 2) for i in range(5)]

        # Initialize positional values
        self.start = (float(sy), float(sx), (sr // self.theta) % self.ti)
        self.goal = (float(gy), float(gx))
        self.lastPosition = (-1, -1, -1)  # Coordinates of ending position (will be within certain tolerance of goal)
        self.success = True

        # Goal threshold:  minimum distance that can be covered in one action step
        self.goal_threshold = gt if gt > 0.0 else (0.51 * self.step)

        # Heuristic weight (set to <= 1.0 for optimal path or 0.0 for Dijkstra)
        self.hw = hw

        # Check to see if start and goal cells lie within map boundaries
        t = self.rad + self.clr
        if not (t <= self.start[0] < self.map.height - t) or not (t <= self.start[1] < self.map.width - t):
            self.error = ValueError("Start lies outside of map boundaries!")
            return
        elif not (t <= self.goal[0] < self.map.height - t) or not (t <= self.goal[1] < self.map.width - t):
            self.error = ValueError("Goal lies outside of map boundaries!")
            return

        # Check to see if start and goal cells are in free spaces
        if self.map.is_colliding((self.start[0], self.start[1]), thickness=self.rad):
            self.error = ValueError("Start lies within obstacle space!")
            return
        if self.map.is_colliding((self.goal[0], self.goal[1]), thickness=self.rad):
            self.error = ValueError("Goal lies within obstacle space!")
            return

        # Define cell maps to track exploration
        self.openList = []  # List of coordinates to be explored, in the form: [(y, x, t), cost, action]
        self.configSpace = np.zeros((int(ceil(self.map.height / self.res)),
                                     int(ceil(self.map.width / self.res)), self.ti), dtype=np.uint8)

        # Grid of cells pending exploration
        self.openGrid = np.zeros_like(self.configSpace)

        # Grid of explored cells
        self.closeGrid = np.zeros_like(self.configSpace, dtype=np.uint8)

        # Grid containing parent cells
        self.parentGrid = np.zeros((self.configSpace.shape[0], self.configSpace.shape[1],
                                    self.configSpace.shape[2], 3), dtype=np.int) - 1
        # Grid containing movement policy
        self.actionGrid = np.zeros((self.configSpace.shape[0], self.configSpace.shape[1],
                                    self.configSpace.shape[2], 3), dtype=np.float64)

    def on_goal(self, point):
        """
        Checks to see whether the given point is on the goal (within a certain threshold).

        Parameters
        ----------
        point: tuple of float
            The point for which to check

        Returns
        -------
        bool
            True if the point is on the goal, False otherwise
        """
        return sqrt((self.goal[0] - point[0]) ** 2 + (self.goal[1] - point[1]) ** 2) <= self.goal_threshold

    def next_position(self, cell, a):
        y, x, t = cell
        t_rad = t * self.theta * pi / 180.0
        a_rad = a * self.theta * pi / 180.0
        if a == 0:
            ny = y + self.step * sin(t_rad)
            nx = x + self.step * cos(t_rad)
        else:
            cr = self.step / a_rad
            cy = y - cr * sin(t_rad - pi / 2.0)
            cx = x - cr * cos(t_rad - pi / 2.0)
            ny = cy + cr * sin(t_rad + a_rad - pi / 2.0)
            nx = cx + cr * cos(t_rad + a_rad - pi / 2.0)
        nt = (t + a) % self.ti
        return ny, nx, nt

    def solve(self):
        """
        Find a path to the goal, if there is a valid path.
        """
        # If there were errors in initialization, exit.
        if self.error is not None:
            return self.error
        # Initialize the open list/grid with the start cell
        # TODO: Convert openList into a numpy array.
        self.openList = [[self.start, 0]]  # [point, cost]
        self.openGrid[int(self.start[0] / self.res), int(self.start[1] / self.res), self.start[2]] = 1
        path_points = []
        logging.debug("Searching for optimal path...")
        explored_count = 0
        while len(self.openList) > 0:
            # Find index of minimum cost cell
            cost_list = []
            for i in range(len(self.openList)):
                # Heuristic is the Euclidean distance to the goal
                open_y = self.openList[i][0][0]
                open_x = self.openList[i][0][1]
                heuristic = sqrt((open_y - self.goal[0]) ** 2 + (open_x - self.goal[1]) ** 2) * self.hw
                cost_list.append(self.openList[i][1] + heuristic)
            index = int(np.argmin(cost_list, axis=0))
            cell = self.openList[index][0]
            cost = self.openList[index][1]

            # See if goal cell has been reached (with threshold condition)
            if self.on_goal(cell):
                self.lastPosition = cell
                self.openList = []

            # Expand cell
            else:
                for a in range(len(self.actions)):
                    next_cell = self.next_position(cell, a - (len(self.actions) // 2))
                    ny, nx, nt = next_cell
                    # theta_norm = int(nt * 180.0 / pi) // self.theta

                    # Check for map boundaries
                    if 0.0 <= ny < self.map.height and 0.0 <= nx < self.map.width:
                        # Check for obstacles
                        collision = self.map.is_colliding((ny, nx, self.rad + self.clr))
                        collision = collision or self.map.is_colliding(((ny + cell[0]) / 2.0, (nx + cell[1]) / 2.0,
                                                                       self.rad + self.clr * 1.5))
                        if not collision:
                            # Check whether cell has been explored
                            if not self.closeGrid[int(ny / self.res), int(nx / self.res), nt]:
                                # Check if cell is already pending exploration
                                if not self.openGrid[int(ny / self.res), int(nx / self.res), nt]:
                                    self.openList.append([(ny, nx, nt), cost + self.step])
                                    parent = [int(cell[0] / self.res), int(cell[1] / self.res), cell[2]]
                                    self.parentGrid[int(ny / self.res), int(nx / self.res), nt] = parent
                                    action = [cell[0], cell[1], cell[2]]
                                    self.actionGrid[int(ny / self.res), int(nx / self.res), nt] = action
                                    self.openGrid[int(ny / self.res), int(nx / self.res), nt] = 1

                self.openList.pop(index)
                if len(self.openList) == 0:
                    self.success = False

            # Mark the cell as having been explored
            self.openGrid[int(cell[0] / self.res), int(cell[1] / self.res), cell[2]] = 0
            self.closeGrid[int(cell[0] / self.res), int(cell[1] / self.res), cell[2]] = 1

            explored_count += 1
            if explored_count % 1000 == 0 and explored_count > 0:
                logging.debug("%d states explored" % explored_count)

        # Check for failure to reach the goal cell
        if self.on_goal(self.start):
            logging.warning("No path generated.  Robot starts at goal space.")
        elif not self.success:
            self.error = Exception("Failed to find a path to the goal!")

        # Backtrack from the goal cell to extract an optimal path.
        else:
            logging.debug("Goal reached!")
            goal_y = int(self.lastPosition[0] / self.res)
            goal_x = int(self.lastPosition[1] / self.res)
            goal_r = self.lastPosition[2]
            current_cell = (goal_y, goal_x, goal_r)
            current_actual = self.lastPosition
            next_cell = tuple(self.parentGrid[current_cell])
            # TODO: Convert path_points into numpy array.
            path_points = []
            while sum(next_cell) >= 0:
                if current_cell[2] == next_cell[2]:
                    mid_y = np.linspace(current_actual[0], self.actionGrid[current_cell][0], self.cs, endpoint=False)
                    mid_x = np.linspace(current_actual[1], self.actionGrid[current_cell][1], self.cs, endpoint=False)
                    for i in range(self.cs):
                        path_points.append([mid_y[i], mid_x[i]])
                else:
                    y, x, t = current_actual
                    t_rad = t * self.theta * pi / 180.0
                    a_rad = (t - next_cell[2]) * self.theta * pi / 180.0
                    a_rad = ((a_rad + pi) % (2.0 * pi)) - pi
                    sweep = ((current_cell[2] - next_cell[2] + (self.ti // 2)) % self.ti) - (self.ti // 2)
                    cr = self.step / (sweep * self.theta * pi / 180.0)
                    cy = y - cr * sin(t_rad - pi / 2.0)
                    cx = x - cr * cos(t_rad - pi / 2.0)
                    for i in range(self.cs):
                        ny = cy + cr * sin(t_rad - a_rad * i / self.cs - pi / 2.0)
                        nx = cx + cr * cos(t_rad - a_rad * i / self.cs - pi / 2.0)
                        path_points.append([ny, nx])
                current_actual = self.actionGrid[current_cell]
                current_cell = next_cell
                next_cell = tuple(self.parentGrid[next_cell])
            path_points.reverse()
        return self.error if self.error is not None else path_points
