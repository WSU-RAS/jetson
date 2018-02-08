#!/usr/bin/env python2
"""
Listen to object locations published from /find_objects and save to Redis DB
for persistence

object_db_service.py provides a service to access these saved locations.
get_object_db.py and get_object_service.py are examples showing how to get the
locations.
"""
import json
import rospy
import redis
from ras_jetson_msgs.msg import Object

class UpdateObjectDBNode:
    """
    Save object locations

    Usage:
        node = UpdateObjectDBNode()
        rospy.spin()
    """
    def __init__(self):
        # Name this node
        rospy.init_node('updateObjectDB')

        # Params
        self.server = rospy.get_param("~server", "localhost")
        self.port   = rospy.get_param("~port", "6379")
        self.prefix = rospy.get_param("~prefix", "object")

        self.redis = redis.StrictRedis(host=self.server, port=self.port, db=0)

        # Listen to object locations that are published
        rospy.Subscriber("/find_objects", Object, self.callback_object)

    def callback_object(self, data):
        """
        Save the object location when we see it
        """
        try:
            # TODO support multiple of the same object
            # Save an array of object locations
            self.redis.set(self.prefix+"_"+data.name, json.dumps([{
                    "name": data.name,
                    "x": data.x,
                    "y": data.y,
                    "z": data.z
                }]))
        except:
            rospy.logerr("Cannot insert row")

if __name__ == '__main__':
    try:
        node = UpdateObjectDBNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
