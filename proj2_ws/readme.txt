ENPM662 - Project 2 - NavBot

How to run this program:
 - Make the catkin package from inside the proj_w2 directory:
    catkin_make

 - Run the following command (which only has to be ran at least once per terminal):
    source devel/setup.bash

 - Run the launch file:
    roslaunch navbot gazebo.launch

 - You can specify additional arguments to spawn in different locations, or run a simulation for placing a cube into the tray. All are valid examples:
    roslaunch navbot gazebo.launch xi:=2 yi:=-5.5 ri:=90
    roslaunch navbot gazebo.launch yi:=3.5 ri=180 mode:=plan
    roslaunch navbot gazebo.launch mode:=cube
    roslaunch --ros-args navbot gazebo.launch

 - The first example spawns NavBot at initial location x=2.0, y=-5.5, and rotated 90 degrees around z-axis. Default location is x=0.0, y=0.0, rotation=0 degrees.
 - The second example spawns NavBot at initial location x=0.0, y=3.5, and rotated 180 degrees around z-axis in "path planning" mode.
 - Valid mode values are "teleop", "plan", "cube". Default mode is "teleop".
 - The third example runs a demo of NavBot moving a cube. Using the cube mode ignores position arguments.
 - The fourth example shows how to use the launch file.

Using teleop:
 - While in teleop, the window running the ROS package must be in focus, so just click this terminal to focus it. At this point, use the WASD keys to move NavBot.
 - [W] moves forward, [S] moves backward, [A] turns left, and [D] turns right.
 - [F] can be used to move the arm into home position, and [G] collapses the arm.
 - Use [H] to print these instructions again.
 - Use [T] to toggle between "teleop" and "plan" modes.

Issuing commands:
 - Using another terminal, you can issue commands to NavBot. Included with this package are helpful scripts for publishing commands that the robot is subscribed to.
 - move_ee.sh takes 3 arguments: the x, y, and z location of the end effector for inverse kinematics. The robot will attempt to move the end effector there if it is within reach. 
    ./move_ee.sh 0 0 1  # Moves the end effector to x=0.0, y=0.0, z=1.0
 - move_joints.sh takes 6 arguments: the robot manipulator joint values for forward kinematics. See the report for details, or just experiment with different values.
    ./move_joints.sh 0 0 0 0 0 0  # Moves to home position
 - move_drive.sh takes 2 arguments: the x and y location to which NavBot will drive using path planning.
    ./move_ee.sh 6.5 -6  # Instructs NavBot to drive to x=6.5, y=-6.0
 - All three scripts are usable in plan and teleop mode. If used in teleop mode, move_drive.sh will not do anything immediately, but will queue goal locations for plan mode.

