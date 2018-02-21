#!/usr/bin/env python2
"""
Example of how to get object locations from the database

rosrun object_detection get_object_db.py
"""
import json
import rospy
import redis

class GetObjectDB:
    """
    Access object locations

    Usage:
        db = ObjectDB()
    """
    def __init__(self):
        # Name this node
        rospy.init_node('getObjectDB')

        # Params
        self.server = rospy.get_param("~server", "localhost")
        self.port   = rospy.get_param("~port", "6379")
        self.prefix = rospy.get_param("~prefix", "object")

        self.redis = redis.StrictRedis(host=self.server, port=self.port, db=0)

    def get(self, name):
        data = self.redis.get(self.prefix+"_"+name)

        if data:
            data = json.loads(data.decode("utf-8"))
        else:
            data = None

        return data

if __name__ == '__main__':
    try:
        node = GetObjectDB()

        # TODO only for testing right now...
        print "Pillbottle:", node.get("pillbottle")

        rospy.spin()
    except rospy.ROSInterruptException:
        pass
