#!/usr/bin/env python

import numpy as np
import sys
from math import pi, sin, cos, exp, sqrt, atan2, erf
import rospy
from std_msgs.msg import Float64, Float64MultiArray
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
import select
import termios
import tty
import time
from multiprocessing import Process, Queue
from astar import AStar
from obstacleMap import ObstacleMap
import geometry


class Robot:
    """
    Attributes
    ----------
    plan_process: Process
        Process running the motion planning algorithm.
    """
    gt = 0.5
    step = 1.5
    cs = 8

    def __init__(self, location, mode, poses):
        x, y, r = location
        self.x = x
        self.y = y
        self.r = r
        self.rad = 0.5
        self.ee_x = 0.0
        self.ee_y = 0.0
        self.ee_z = 0.0
        self.links = np.array([0.48, 0.4, 0.4, 0.345, 0.118])
        self.mode = mode
        self.collapse_pose = np.array([[0, 90, -78, 168, 0, 0]]) * pi / 180.0
        if poses is None:
            self.poses = np.zeros((1, 6))
            if self.mode == "plan":
                self.poses = np.append(self.poses, self.collapse_pose, axis=0)
        else:
            self.poses = np.zeros((len(poses) + 1, 6))
            self.poses[1:, :] = np.array([self.inv_pose(poses[i, 0],
                                                        poses[i, 1],
                                                        poses[i, 2]).tolist() for i in range(len(poses))])
        self.t_pose = 0.0

        # Motion planning
        self.goals = np.zeros((0, 2))
        self.is_on_path = False
        self.plan_process = None
        self.queue = Queue()
        self.path = np.zeros((0, 2), dtype=np.float64)
        self.path_tangents = np.zeros((0,), dtype=np.float64)
        self.rev_i = -1
        self.ww = 20.0
        self.wh = 20.0
        self.polygons = [geometry.Polygon([[0.0, 5.85],  # Bottom left wall
                                           [12.0, 5.85],
                                           [12.0, 6.15],
                                           [0.0, 6.15]]),
                         geometry.Polygon([[8.0, 13.85],  # Top right wall
                                           [20.0, 13.85],
                                           [20.0, 14.15],
                                           [8.0, 14.15]]),
                         geometry.Polygon([[2.5, 2.2],  # Bottom left table
                                           [3.5, 2.2],
                                           [3.5, 3.8],
                                           [2.5, 3.8]]),
                         geometry.Polygon([[16.5, 16.2],  # Top right table
                                           [17.5, 16.2],
                                           [17.5, 17.8],
                                           [16.5, 17.8]])]

        # Control
        self.steer = 0.0
        self.accel = 0.0
        self.max_steer = pi / 4.0
        self.max_drive = 20.0
        self.drive_accel = self.max_drive / 2.0
        self.turn_accel = self.max_steer / 2.0

        # Visualization
        self.vis_z = 0.1
        self.rviz_goal_markers = MarkerArray()
        self.rviz_path = Path()
        self.rviz_path.header.frame_id = "/map"

    def add_goal(self, gx, gy):
        self.goals = np.append(self.goals, np.array([[gx, gy]]), axis=0)
        marker = Marker()
        marker.header.frame_id = "/map"
        marker.header.stamp = rospy.Time.now()
        marker.id = len(self.rviz_goal_markers.markers)
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = gx
        marker.pose.position.y = gy
        marker.pose.position.z = self.vis_z
        marker.pose.orientation.z = 1.0
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color.a = 1.0
        marker.color.r = 0.65
        marker.color.g = 0.0
        marker.color.b = 1.0
        self.rviz_goal_markers.markers.append(marker)

    def pop_goal(self):
        self.goals = np.delete(self.goals, 0, axis=0)
        self.rviz_goal_markers.markers[0].action = Marker.DELETE

    def set_path(self, path):
        self.path = path
        delta_x = self.path[1:, 0] - self.path[:-1, 0]
        delta_y = self.path[1:, 1] - self.path[:-1, 1]
        self.path_tangents = np.arctan2(delta_y, delta_x)
        self.rviz_path.poses = []
        for i in range(self.path.shape[0]):
            next_pose = PoseStamped()
            next_pose.header = self.rviz_path.header
            next_pose.pose.position.x = self.path[i, 0]
            next_pose.pose.position.y = self.path[i, 1]
            next_pose.pose.position.z = self.vis_z
            next_pose.pose.orientation.z = 1.0
            self.rviz_path.poses.append(next_pose)

    def reset_path(self):
        self.path = np.zeros((0, 2))
        self.rviz_path.poses = []

    def update(self):
        # Are there goals in the queue?
        if self.goals.shape[0] > 0:
            # Is the robot currently following an assigned path?
            if self.is_on_path:
                # Are there multiple points in the path?
                if self.path.shape[0] >= 2:
                    # Is the robot close enough to the end of the path?
                    if sqrt((self.x - self.path[-1, 0])**2 + (self.y - self.path[-1, 1])**2) <= Robot.gt:
                        self.reset_path()
                        self.is_on_path = False
                        self.pop_goal()
                        self.steer = 0.0
                        self.accel = 0.0
                    # Follow the assigned path
                    else:
                        # Is the robot moving in reverse?
                        if self.is_reversing():
                            # Is the robot close enough to the point at which it should start moving forward?
                            mid_x, mid_y = self.path[self.rev_i]  # Reversal point
                            if sqrt((self.x - mid_x)**2 + (self.y - mid_y)**2) <= Robot.gt * 0.5:
                                self.rev_i = -1
                        move_dir = self.r + (pi if self.is_reversing() else 0.0)
                        segment = Robot.step / Robot.cs
                        distances = (self.path[:-1, 0] - self.x) ** 2 + (self.path[:-1, 1] - self.y) ** 2  # Squared
                        distances += abs(((self.path_tangents - move_dir) + pi) % (2.0 * pi) - pi) / pi * 0.5
                        nearest_index = int(np.argmin(distances))
                        nx, ny = self.path[nearest_index]
                        nearest_distance = sqrt((ny - self.y) ** 2 + (nx - self.x) ** 2)
                        tangent_dir = ((self.path_tangents[nearest_index] - move_dir) + pi) % (2.0 * pi) - pi
                        # Did the robot stray too far from the path?
                        if nearest_distance > 1.5:
                            self.reset_path()
                            self.is_on_path = False
                        else:
                            # Is the robot approximately on the line?
                            if nearest_distance >= segment / 2.0:
                                dir_to_path = (atan2(ny - self.y, nx - self.x) - move_dir + pi) % (2.0 * pi) - pi
                                decay_rate = 1.25
                                dir_weight = 1.0 - exp(-decay_rate * nearest_distance)
                                vec1 = (cos(dir_to_path) * dir_weight, sin(dir_to_path) * dir_weight)
                                vec2 = (cos(tangent_dir) * (1.0 - dir_weight), sin(tangent_dir) * (1.0 - dir_weight))
                                target_dir = atan2(vec1[1] + vec2[1], vec1[0] + vec2[0])
                            else:
                                target_dir = tangent_dir
                            self.steer = max(-self.max_steer, min(self.max_steer, target_dir))
                            if self.is_reversing():
                                self.steer = 0.0
                            self.accel = -0.5 if self.is_reversing() else 1.0
            # Is the robot planning a path?
            elif self.plan_process is not None:
                # Is planning algorithm complete?
                if not self.plan_process.is_alive():
                    result, duration, reverse_index = self.queue.get()
                    rospy.loginfo("Path planner duration: %.4f seconds" % duration)
                    # Was path invalid?
                    if isinstance(result, Exception):
                        rospy.logwarn(result)
                        self.pop_goal()
                    # Was goal different from start?
                    elif len(result) > 1:
                        self.is_on_path = True
                        rospy.loginfo("Found valid path!")
                        new_path = np.array(result)
                        new_path[:, 0] = new_path[:, 0] - self.ww/2.0
                        new_path[:, 1] = new_path[:, 1] - self.wh/2.0
                        self.set_path(new_path)
                        self.rev_i = reverse_index
                        self.is_on_path = True
                    else:
                        rospy.loginfo("NavBot started on goal.")
                        self.pop_goal()
                    self.plan_process = None
                # Wait for planning algorithm to complete
                else:
                    self.steer = 0.0
                    self.accel = 0.0
            # Start planning a path, since there are goals in the queue
            else:
                rospy.loginfo("Planning path to <x=%.2f, y=%.2f>" % (self.goals[0, 0], self.goals[0, 1]))
                self.queue = Queue()
                self.plan_process = Process(target=self.plan_path,
                                            args=(self.x + self.ww/2.0, self.y + self.wh/2.0, self.r, self.rad,
                                                  self.goals[0, 0] + self.ww/2.0, self.goals[0, 1] + self.wh/2.0,
                                                  Robot.gt * 0.75, self.ww, self.wh, self.polygons, self.queue),
                                            name="PathPlaner")
                self.plan_process.start()
        # Idle, since there are no goals in the queue
        else:
            self.steer = 0.0
            self.accel = 0.0

    def is_reversing(self):
        return self.rev_i >= 0

    def odom_callback(self, msg):
        try:
            i = next(j for j in range(len(msg.name)) if msg.name[j] == "NavBot")
            pose = msg.pose[i]  # The variables "name" and "twist" are also available attributes besides "pose".
            self.x = pose.position.x
            self.y = pose.position.y
            self.r = ((atan2(pose.orientation.z, pose.orientation.w) * 2.0 + pi) % (2.0 * pi)) - pi
        except StopIteration:
            pass

    def inv_pose(self, ee_xi, ee_yi, ee_zi):
        # Calculate inverse kinematics using FABRIK
        ee_x = ee_xi - self.x
        ee_y = ee_yi - self.y
        ee_z = ee_zi - self.links[0]

        # Check to see if target end effector location is within reach
        chassis_height = 0.39
        total_distance = sqrt(ee_x ** 2 + ee_y ** 2 + ee_z ** 2)
        link_lengths = self.links[1:4] + np.array([0.0, 0.0, self.links[4]])
        ee_in_chassis = ((sqrt(ee_x**2 + ee_y**2) < self.rad) and (ee_zi < chassis_height))
        if total_distance > np.sum(link_lengths) + 0.18:
            rospy.logwarn("Cannot reach end effector location!")
            return None
        if ee_in_chassis:
            rospy.logwarn("End effector location is probably within chassis!")
            return None

        # If position is within reach, then we may continue
        ee_p = pi / 2.0
        inverse_pose = np.zeros((6,), dtype=np.float64)
        theta_o = atan2(ee_y, ee_x) - self.r
        inverse_pose[0] = ((theta_o + pi) % (2.0 * pi)) - pi
        inverse_pose[1:4] = fabrik(np.array([0, pi/2, 0]), (sqrt(ee_x**2 + ee_y**2), ee_z, ee_p), link_lengths)
        inverse_pose[2] += pi / 2
        inverse_pose[4] = 0.0
        return inverse_pose

    def ee_callback(self, msg):
        if len(msg.data) != 3:
            rospy.logwarn("Invalid end-effector location: %s" % str(msg.data))
            return
        rospy.loginfo("Provided end effector location <x=%.2f, y=%.2f, z=%.2f>" % msg.data)
        ee_xi, ee_yi, ee_zi = msg.data
        inverse_pose = self.inv_pose(ee_xi, ee_yi, ee_zi)
        if inverse_pose is None:
            return
        rospy.loginfo("Joint locations: %s" % (", ".join(("%d" % int(val * 180.0 / pi)) for val in inverse_pose)))
        self.poses = np.append(self.poses, np.expand_dims(inverse_pose, axis=0), axis=0)

    def joint_callback(self, msg):
        if len(msg.data) != 6:
            rospy.logwarn("Invalid joint configuration: %s" % str(msg.data))
            return
        rospy.loginfo("Provided joint values <%d, %d, %d, %d, %d, %d>" % msg.data)
        joints = np.array([element for element in msg.data], dtype=np.float64) * pi / 180.0
        joints = np.mod(joints + pi, 2.0*pi) - pi

        # If position is within reach, then we may continue
        frames = [dh_trans(-pi/2.0,     joints[0] + pi,             0.0, self.links[0]),
                  dh_trans(0.0,     joints[1] - pi/2.0,   self.links[1],           0.0),
                  dh_trans(0.0,     joints[2] - pi/2.0,   self.links[2],           0.0),
                  dh_trans(-pi/2.0,          joints[3],   self.links[3],           0.0),
                  dh_trans(-pi/2.0, joints[4] - pi/2.0,             0.0,           0.0),
                  dh_trans(0.0,              joints[5],             0.0, self.links[4])]
        transformation_matrix = final_trans(frames)
        base_rot_trans = np.array([[cos(self.r),  -sin(self.r),  0.0,  0.0],
                                   [sin(self.r),   cos(self.r),  0.0,  0.0],
                                   [0.0,                   0.0,  1.0,  0.0],
                                   [0.0,                   0.0,  0.0,  1.0]])
        base_origin_trans = np.array([[1.0, 0.0, 0.0, self.x],
                                      [0.0, 1.0, 0.0, self.y],
                                      [0.0, 0.0, 1.0,    0.0],
                                      [0.0, 0.0, 0.0,    1.0]])
        ee_transform = np.matmul(np.matmul(base_rot_trans, transformation_matrix), base_origin_trans)
        ee_origin = ee_transform[0:3, 3]
        rospy.loginfo("End-effector location: <x=%.2f, y=%.2f, z=%.2f>" % tuple([val for val in ee_origin]))
        self.poses = np.append(self.poses, np.expand_dims(joints, axis=0), axis=0)

    def drive_callback(self, msg):
        if len(msg.data) != 2:
            rospy.logwarn("Invalid goal configuration: %s" % str(msg.data))
            return
        gx, gy = msg.data
        self.add_goal(gx, gy)

    @staticmethod
    def plan_path(rx, ry, rot, rad, gx, gy, gt, ww, wh, polygons, queue):
        """
        Find a path to the goal, if there is a feasible path.

        Parameters
        ----------
        rx: float
            X position of the robot's initial state for the planner.
        ry: float
            Y position of the robot's initial state for the planner.
        rot: float
            Orientation, in radians, of the robot's initial state for the planner.
        rad: float
            Radius of the robot.
        gx: float
            X position of the goal for the planner.
        gy: float
            Y position of the goal for the planner.
        gt: float
            Goal threshold.
        ww: float
            World width
        wh: float
            World height
        polygons: list
            List of polygons with which to construct an obstacle map for the configuration space.
        queue: Queue
            Queue on which to place the final path.
        """
        start_time = time.time()
        wall_time = 0.2  # Seconds. A path-planning duration greater than this means there's likely not a wall ahead.
        reverse_index = -1  # Index of path point at which the robot should reverse direction
        obstacle_map = ObstacleMap(wh, ww, rad, polygons=polygons)
        planner = AStar(ry, rx, int(round(rot * 180.0 / pi)), gy, gx, rad=rad, d_theta=30,
                        step=Robot.step, hw=1.2, gt=gt, cell_split=Robot.cs, obstacle_map=obstacle_map)
        result = planner.solve()

        # If first iteration is successful (or ValueError, which means invalid initial conditions), return path here
        if not isinstance(result, Exception):
            result = [[x, y] for y, x in result]
        else:
            if not isinstance(result, ValueError):
                # Could there be a wall just ahead of the robot?
                if time.time() - start_time < wall_time:
                    # See if the robot can reverse
                    reverse_planner = AStar(ry, rx, int(round(rot * 180.0 / pi + 180.0)),
                                            ry - Robot.step * sin(rot), rx - Robot.step * cos(rot),
                                            rad=rad, d_theta=30, step=1.5 * Robot.step, hw=1.5, gt=gt * 0.5,
                                            cell_split=Robot.cs, obstacle_map=obstacle_map)
                    reverse_result = reverse_planner.solve()
                    # Was robot able to reverse?
                    if not isinstance(reverse_result, Exception):
                        reverse_result = [[x, y] for y, x in reverse_result]
                        new_rx, new_ry = reverse_result[-1]
                        planner = AStar(new_ry, new_rx, int(round(rot * 180.0 / pi)), gy, gx, rad=rad, d_theta=30,
                                        step=Robot.step, hw=1.2, gt=gt, cell_split=Robot.cs, obstacle_map=obstacle_map)
                        new_result = planner.solve()
                        # After reversing, could the robot then find a feasible path?
                        if not isinstance(new_result, Exception):
                            result = [[x, y] for x, y in reverse_result]
                            result.extend([[x, y] for y, x in new_result])
                            reverse_index = len(reverse_result) - 1
        duration = time.time() - start_time
        queue.put((result, duration, reverse_index))


def get_key(settings):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def dh_trans(alpha, theta, a, d):
    """
    Calculate transformation matrix from provided Denavit-Hartenberg (DH) parameters.

    Parameters
    ----------
    alpha: float
        Angle required to rotate around new x-axis so that old z-axis points in direction of new z-axis.
    theta: float
        Angle required to rotate around old z-axis so that old x-axis points in direction of new x-axis.
    a: float
        Perpendicular distance between old z-axis and new z-axis.
    d: float
        Distance along old z-axis by which new joint origin is shifted.

    Returns
    -------
    np.ndarray
        Transformation matrix calculated by DH parameters.
    """
    frame = np.array([[cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta)],
                      [sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
                      [0.0,                    sin(alpha),             cos(alpha),            d],
                      [0.0,                           0.0,                    0.0,          1.0]])
    return frame


def final_trans(frames):
    final_frame = frames[0]
    for i in range(len(frames) - 1):
        final_frame = np.matmul(final_frame, frames[i+1])
    return final_frame


def fabrik(joint_i, ee, d):
    n = len(joint_i)
    ee_x, ee_y, ee_r = ee

    # Calculate initial joint positions from initial joint angles
    pos = np.zeros((n + 1, 2))
    for i in range(n):
        pos[i+1, 0] = pos[i, 0] + d[i]*sin(np.sum(joint_i[:i+1]))
        pos[i+1, 1] = pos[i, 1] + d[i]*cos(np.sum(joint_i[:i+1]))

    # Main loop
    epsilon = 0.01
    diff1 = diff2 = epsilon * 2.0
    iterations = 0
    iteration_limit = 100
    while (diff1 > epsilon or diff2 > epsilon) and iterations < iteration_limit:
        # Iterate forward
        pos[n] = np.array([ee_x, ee_y])  # Move end effector to target location
        an_x, an_y = pos[0]  # Anchor position is located at first joint
        for i in range(n):
            cx, cy = pos[n-i]  # Current x and y
            nx, ny = pos[n-i-1]  # Next x and y
            angle = atan2(ny - cy, nx - cx)  # Direction from current joint to next joint
            dist = d[n-i-1]
            pos[n-i-1] = np.array([cx + dist*cos(angle), cy + dist*sin(angle)])
        diff1 = sqrt((an_x - pos[0, 0])**2 + (an_y - pos[0, 1])**2)

        # Iterate backward
        pos[0] = np.zeros((2,))  # Move first joint to its initial location (or other location as desired)
        an_x, an_y = pos[n]  # Anchor position is located at end effector
        for i in range(n):
            cx, cy = pos[i]  # Current x and y
            nx, ny = pos[i+1]  # Next x and y
            angle = atan2(ny - cy, nx - cx)  # Direction from current joint to next joint
            dist = d[i]
            pos[i+1] = np.array([cx + dist*cos(angle), cy + dist*sin(angle)])
        diff2 = sqrt((an_x - pos[n, 0])**2 + (an_y - pos[n, 1])**2)
        iterations += 1
    joint_f_abs = np.arctan2(pos[1:n + 1, 1] - pos[:n, 1], pos[1:n + 1, 0] - pos[:n, 0])  # For absolute angles
    joint_f = np.copy(joint_f_abs)
    joint_f[0] -= (pi/2.0 + joint_i[0])
    for i in range(n-1):
        joint_f[i+1] = ((joint_f_abs[i+1] - joint_f_abs[i] + pi) % (2.0 * pi)) - pi
    return joint_f


def main(args):
    # ROS init node
    rospy.init_node("navbot_demo")
    settings = termios.tcgetattr(sys.stdin)

    # Print system arguments
    rospy.logdebug("ARGUMENTS:")
    for i in range(len(args)):
        rospy.logdebug("Arg %i: %s" % (i, args[i]))

    # Create robot instance
    operation_mode = args[3] if args[3] in ["teleop", "plan", "cube"] else "teleop"
    poses = np.array([[7.3, 7, 0.8],
                      [7.4, 6.5, 0.55],
                      [7.2, 6.5, 0.55],
                      [7.4, 6.5, 0.55],
                      [7.4, 6.95, 0.55],
                      [7.1, 6.95, 0.55],
                      [7.1, 6.42, 0.55]])
    robot = Robot([float(arg) for arg in args[0:3]], operation_mode, poses if operation_mode == "cube" else None)

    # Subscribe to transformation information, end-effector location, and joint positions
    rospy.Subscriber("/gazebo/model_states", ModelStates, robot.odom_callback)
    if robot.mode != "cube":
        rospy.Subscriber("/ee_location", Float64MultiArray, robot.ee_callback)
        rospy.Subscriber("/joint_positions", Float64MultiArray, robot.joint_callback)
        rospy.Subscriber("/drive_location", Float64MultiArray, robot.drive_callback)

    # Create a topic for each controller.
    pub_rr = rospy.Publisher('/navbot_control/controller_rear/command', Float64, queue_size=10)
    pub_tl = rospy.Publisher('/navbot_control/controller_f_l_wheel/command', Float64, queue_size=10)
    pub_tr = rospy.Publisher('/navbot_control/controller_f_r_wheel/command', Float64, queue_size=10)
    pub_j1 = rospy.Publisher('/navbot_control/controller_joint_1/command', Float64, queue_size=10)
    pub_j2 = rospy.Publisher('/navbot_control/controller_joint_2/command', Float64, queue_size=10)
    pub_j3 = rospy.Publisher('/navbot_control/controller_joint_3/command', Float64, queue_size=10)
    pub_j4 = rospy.Publisher('/navbot_control/controller_joint_4/command', Float64, queue_size=10)
    pub_j5 = rospy.Publisher('/navbot_control/controller_joint_5/command', Float64, queue_size=10)
    pub_fl = rospy.Publisher('/navbot_control/controller_f_l_rim/command', Float64, queue_size=10)
    pub_fr = rospy.Publisher('/navbot_control/controller_f_r_rim/command', Float64, queue_size=10)
    pub_ee = rospy.Publisher('/navbot_control/controller_joint_end_effector/command', Float64, queue_size=10)

    # RViz Publishers
    pub_rviz_path = rospy.Publisher('/nav_msgs/Path', Path, queue_size=100)
    pub_rviz_marker = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=100)

    # Teleop
    help_message = """
        Control NavBot!
        ---------------------------
        TAP the following keys to move.
        \"[GO ([W]), STOP ([S]), LEFT ([A]), RIGHT ([D])]\"

        Moving around:
                [W]
          [A]   [S]   [D]
        [F] for manipulator home position
        [G] for manipulator collapsed position
        CTRL-C to quit
        [T] to toggle mode between teleop and plan
        [H] to repeat this message
        """
    if robot.mode != "cube":
        rospy.logwarn(help_message)
    key_binds = {"forward": "w",
                 "right": "d",
                 "left": "a",
                 "backward": "s",
                 "home": "f",
                 "collapse": "g",
                 "toggle": "t",
                 "help": "h"}
    control_speed = 0.0
    control_turn = 0.0

    # Timing
    t_pose = 0.0
    rate = rospy.Rate(30)
    rospy.sleep(rospy.Duration(3))
    ti = rospy.Time.now().to_sec()
    is_moving_arm = False

    # Poses
    pose_period = 2.0 if robot.mode == "cube" else 3.0  # Duration (in seconds) of pose transition
    e_range = 2.5  # Value at which erf(x) is approximately 1.0

    try:
        while True:
            t = rospy.Time.now().to_sec() - ti

            # Process keyboard inputs
            key = get_key(settings)

            # Toggle
            if key == key_binds["toggle"] and robot.mode != "cube":
                robot.mode = "teleop" if robot.mode == "plan" else "plan"
                if robot.mode == "teleop":
                    robot.path = np.zeros((0, 2))
                    if robot.plan_process is not None:
                        if robot.plan_process.is_alive():
                            robot.plan_process.terminate()
                            robot.plan_process = None
                    robot.is_on_path = False
                elif robot.mode == "plan":
                    robot.poses = np.append(robot.poses, robot.collapse_pose, axis=0)
                rospy.loginfo("Switching to %s" % (robot.mode.upper()))

            # Print help
            elif key == key_binds["help"]:
                rospy.logwarn(help_message)

            # Terminate the program
            if key == '\x03':
                if robot.plan_process is not None:
                    if robot.plan_process.is_alive():
                        robot.plan_process.terminate()
                break

            # Teleop
            if robot.mode == "teleop":
                if key == key_binds["home"]:
                    robot.poses = np.append(robot.poses, np.zeros((1, 6)), axis=0)
                elif key == key_binds["collapse"]:
                    robot.poses = np.append(robot.poses, robot.collapse_pose, axis=0)
                elif key == key_binds["forward"]:
                    control_speed = min(robot.max_drive, control_speed + robot.drive_accel)
                elif key == key_binds["backward"]:
                    control_speed = max(-robot.max_drive / 2.0, control_speed - robot.drive_accel)
                elif key == key_binds["right"]:
                    control_turn = min(robot.max_steer, control_turn + robot.turn_accel)
                elif key == key_binds["left"]:
                    control_turn = max(-robot.max_steer, control_turn - robot.turn_accel)
                if abs(control_speed) <= 0.0001:
                    control_speed = 0.0
                if abs(control_turn) <= 0.0001:
                    control_turn = 0.0

                pub_rr.publish(-control_speed * 0.5)
                pub_fl.publish(control_speed)
                pub_fr.publish(-control_speed)
                pub_tl.publish(-control_turn)
                pub_tr.publish(-control_turn)

            # Path planning
            elif robot.mode == "plan":
                robot.update()
                pub_rr.publish(-robot.max_drive * robot.accel)
                pub_fl.publish(robot.max_drive * robot.accel)
                pub_fr.publish(-robot.max_drive * robot.accel)
                pub_tl.publish(robot.steer)
                pub_tr.publish(robot.steer)

            # Keep the robot still (such as during the cube demo)
            else:
                pub_rr.publish(0.0)
                pub_fl.publish(0.0)
                pub_fr.publish(0.0)
                pub_tl.publish(0.0)
                pub_tr.publish(0.0)

            # Move the manipulator, if there are poses in the queue
            if robot.poses.shape[0] > 1:
                if not is_moving_arm:
                    t_pose = t
                is_moving_arm = True
                joints = np.zeros((6,), dtype=np.float64)
                alpha = erf(2*e_range/pose_period * (t - t_pose - pose_period/2))
                for i in range(joints.shape[0]):
                    joints[i] = (robot.poses[1][i] - robot.poses[0][i])/2.0 * alpha
                    joints[i] += (robot.poses[0][i] + robot.poses[1][i])/2.0
                if t - t_pose >= pose_period:
                    robot.poses = np.delete(robot.poses, 0, axis=0)
                    t_pose = t
                val1 = joints[0]
                val2 = joints[1]
                val3 = joints[2]
                val4 = joints[3]
                val5 = joints[4]
                val_ee = joints[5]
                pub_j1.publish(val1)
                pub_j2.publish(val2)
                pub_j3.publish(val3)
                pub_j4.publish(val4)
                pub_j5.publish(val5)
                pub_ee.publish(val_ee)
            else:
                is_moving_arm = False
                if robot.mode == "cube":
                    rospy.sleep(rospy.Duration(5))
                    break

            # RViz visualization
            # Goals for motion planning
            for i in range(len(robot.rviz_goal_markers.markers)):
                robot.rviz_goal_markers.markers[i].color.r = 0.65 + 0.3 * sin(t * pi)
            pub_rviz_marker.publish(robot.rviz_goal_markers)
            if len(robot.rviz_goal_markers.markers) > 0:
                if robot.rviz_goal_markers.markers[0].action == Marker.DELETE:
                    robot.rviz_goal_markers.markers.pop(0)
                    for i in range(len(robot.rviz_goal_markers.markers)):
                        robot.rviz_goal_markers.markers[i].id = i
            # Path from motion planner
            pub_rviz_path.publish(robot.rviz_path)

            rate.sleep()

    # Log exceptions
    except Exception as EE:
        rospy.logerr("EXCEPTION: %s" % EE)

    # End gracefully
    finally:
        if robot.plan_process is not None:
            if robot.plan_process.is_alive():
                robot.plan_process.terminate()
        rospy.loginfo("Simulation terminated. Cleaning processes...")
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


if __name__ == '__main__':
    main(sys.argv[1:])
