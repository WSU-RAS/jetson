#!/usr/bin/env python2
"""
Go To Object node

This node listens to /go_to where you can publish an object name to tell the
robot to go to the last location it saw this object.

Example:
    rostopic pub -1 /go_to std_msgs/String "glass"
"""
import json
import rospy
import actionlib
from std_msgs.msg import String
from ras_jetson.srv import ObjectQuery, ObjectQueryResponse
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus

def getObjectLocation(name):
    """
    Get object location from object name
    """
    rospy.wait_for_service("query_objects")

    try:
        query = rospy.ServiceProxy("query_objects", ObjectQuery)
        results = query(name)
        return results.locations
    except rospy.ServiceException, e:
        rospy.roserr("Service call failed: %s" % e)

    return None

class GoToObject:
    """
    Receive messages to go to object

    Usage:
        node = GoToObjectNode()
        rospy.spin()
    """
    def __init__(self):
        self.running = False

        # Name this node
        rospy.init_node('goToObject')

        # Cancel on shutdown
        rospy.on_shutdown(self.shutdown)

        # Tell the action client that we want to spin a thread by default
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)

        # Allow up to 5 seconds for the action server to come up
        self.move_base.wait_for_server(rospy.Duration(5))
        rospy.loginfo("Waiting for the action server")

        # Listen to object locations that are published
        rospy.Subscriber("/go_to", String, self.callback_object)

    def goTo(self, x, y):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.w = 1.0 # Go forward

        # Start moving
        self.running = True
        self.move_base.send_goal(goal)

        # Allow TurtleBot up to 60 seconds to complete task
        success = self.move_base.wait_for_result(rospy.Duration(600))
        self.running = False

        if not success:
            self.move_base.cancel_goal()
            rospy.loginfo("Move base goal failed")
        else:
            # We made it!
            state = self.move_base.get_state()

            if state == GoalStatus.SUCCEEDED:
                rospy.loginfo("Move base goal succeeded")
            else:
                rospy.logwarn("Move base goal not success")

    def callback_object(self, data):
        name = data.data
        locations = getObjectLocation(name)

        if locations and len(locations) > 0:
            location = locations[0]
            rospy.loginfo("Going to object "+name+" { x: " + str(location.x) + ", y: " + str(location.y) + " }")
            self.goTo(location.x, location.y)
        else:
            rospy.logerr("Cannot go to object "+name)

    def shutdown(self):
        rospy.loginfo("Stopping /go_to")

        # Cancel if we are currently going somewhere
        if self.running:
            self.move_base.cancel_goal()

if __name__ == '__main__':
    try:
        node = GoToObject()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
