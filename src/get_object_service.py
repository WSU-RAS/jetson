#!/usr/bin/env python2
"""
Example of how to get object locations from the Object DB Service

rosrun object_detection get_object_service.py pillbottle
"""
import sys
import rospy
from object_detection_msgs.srv import ObjectQuery, ObjectQueryResponse
import datetime
import dateutil.parser

def getObjectTime(timestamp):
    """
    Convert the .isoformat() back into a datetime object
    https://stackoverflow.com/a/28334064/2698494
    """
    return dateutil.parser.parse(timestamp)

def getObjectLocation(name):
    rospy.wait_for_service("query_objects")

    try:
        query = rospy.ServiceProxy("query_objects", ObjectQuery)
        results = query(name)
        locations = results.locations

        """
        for l in locations:
            if datetime.datetime.now() - getObjectTime(l.time) > datetime.timedelta(seconds=10):
                print "Greater than 10!"
            else:
                print "Less than 10!"
        """

        return locations
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
