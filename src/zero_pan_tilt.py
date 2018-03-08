#!/usr/bin/env python2
"""
Sets the head pan/tilt to a particular position

This is used to initialize it on start. Later on maybe it will track the
person, but for now it'll be fixed.

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

    # Set it once then exit
    traj.add_point([0.0, 0.7], 1.0) # pan, tilt, time to get there
    traj.start()
    traj.wait(5.0)

if __name__ == "__main__":
    main()
