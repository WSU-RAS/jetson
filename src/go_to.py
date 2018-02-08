#!/usr/bin/env python2
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
        # Name this node
        rospy.init_node('goToObject')

        # Listen to object locations that are published
        rospy.Subscriber("/go_to", String, self.callback_object)

    def goTo(self, x, y):
        #tell the action client that we want to spin a thread by default
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        rospy.loginfo("wait for the action server to come up")
        #allow up to 5 seconds for the action server to come up
        self.move_base.wait_for_server(rospy.Duration(5))

        #we'll send a goal to the robot to move 3 meters forward
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.w = 1.0 #go forward

        #start moving
        self.move_base.send_goal(goal)

        #allow TurtleBot up to 60 seconds to complete task
        success = self.move_base.wait_for_result(rospy.Duration(600))

        if not success:
            self.move_base.cancel_goal()
            rospy.loginfo("The base failed to move forward 3 meters for some reason")
        else:
            # We made it!
            state = self.move_base.get_state()

            if state == GoalStatus.SUCCEEDED:
                rospy.loginfo("Hooray, the base moved 3 meters forward")

    def callback_object(self, data):
        name = data.data
        locations = getObjectLocation(name)

        if locations and len(locations) > 0:
            location = locations[0]
            rospy.loginfo("Going to object "+name+" x: " + str(location.x) + " y: " + str(location.y))
            self.goTo(location.x, location.y)
        else:
            rospy.logerr("Cannot go to object "+name)

if __name__ == '__main__':
    try:
        node = GoToObject()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
