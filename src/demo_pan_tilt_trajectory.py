#!/usr/bin/env python2
"""
Example code to send a trajectory command to the pan-tilt

This will make the camera pan/tilt perform some back and forth motion for
testing that the model correctly follows the real movement in rviz.

Based on:
http://sdk.rethinkrobotics.com/wiki/Joint_Trajectory_Client_-_Code_Walkthrough
"""
import sys
import rospy
import argparse
import actionlib
from copy import copy
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, \
        FollowJointTrajectoryGoal
 
class Trajectory(object):
    def __init__(self):
        ns = 'head_controller/'
        self._client = actionlib.SimpleActionClient(
            ns + "follow_joint_trajectory",
            FollowJointTrajectoryAction,
        )
        self._goal = FollowJointTrajectoryGoal()
        server_up = self._client.wait_for_server(timeout=rospy.Duration(10.0))
        if not server_up:
            rospy.logerr("Timed out waiting for Joint Trajectory"
                         " Action Server to connect. Start arbotix node.")
            rospy.signal_shutdown("Timed out waiting for Action Server")
            sys.exit(1)
        self.clear()
    def add_point(self, positions, time):
        point = JointTrajectoryPoint()
        point.positions = copy(positions)
        point.time_from_start = rospy.Duration(time)
        self._goal.trajectory.points.append(point)
    def start(self):
        self._goal.trajectory.header.stamp = rospy.Time.now()
        self._client.send_goal(self._goal)
    def stop(self):
        self._client.cancel_goal()
    def wait(self, timeout=15.0):
        self._client.wait_for_result(timeout=rospy.Duration(timeout))
    def result(self):
        return self._client.get_result()
    def clear(self):
        self._goal = FollowJointTrajectoryGoal()
        self._goal.trajectory.joint_names = ["head_" + joint + "_joint" for joint in \
            ["pan", "tilt"]]
def main():
    rospy.init_node('pan_tilt')
    traj = Trajectory()
    rospy.on_shutdown(traj.stop)

    while not rospy.is_shutdown():
        traj.add_point([0.5, 1.0], 5.0)
        traj.add_point([-0.5, 0.0], 10.0)
        traj.add_point([-0.5, 0.0], 15.0)
        traj.start()
        traj.wait(15.0)
 
if __name__ == "__main__":
    main()
