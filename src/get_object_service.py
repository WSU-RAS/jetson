#!/usr/bin/env python2
"""
Example of how to get object locations from the Object DB Service

rosrun ras_jetson get_object_service.py pillbottle
"""
import sys
import rospy
from ras_jetson.srv import ObjectQuery, ObjectQueryResponse

def getObjectLocation(name):
    rospy.wait_for_service("objectDBService")

    try:
        query = rospy.ServiceProxy("objectDBService", ObjectQuery)
        results = query(name)
        return results.locations
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e

if __name__ == '__main__':
    if len(sys.argv) == 2:
        name = sys.argv[1]
        print "Location of \"%s\":" % name
        print getObjectLocation(name)
    else:
        print "Usage: rosrun ras_jetson get_object_service.py [object]"
