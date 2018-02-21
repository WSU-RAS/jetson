#!/usr/bin/env python2
"""
Example of how to get object locations from the Object DB Service

rosrun object_detection get_object_service.py pillbottle
"""
import sys
import rospy
from object_detection.srv import ObjectQuery, ObjectQueryResponse

def getObjectLocation(name):
    rospy.wait_for_service("query_objects")

    try:
        query = rospy.ServiceProxy("query_objects", ObjectQuery)
        results = query(name)
        return results.locations
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e

if __name__ == '__main__':
    rospy.init_node("get_object_demo")

    if len(sys.argv) == 2:
        name = sys.argv[1]
        print "Location of \"%s\":" % name
        print getObjectLocation(name)
    else:
        print "Usage: rosrun object_detection get_object_service.py [object]"
