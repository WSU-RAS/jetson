#!/usr/bin/env python2
"""
Provide a service for querying object locations saved in Redis DB

Access via:
    /query_objects

See the get_object_service.py example code for how to use this. Or, to test,
you can use:
    rosservice call /query_objects pillbottle
"""
import json
import rospy
import redis
from object_detection_msgs.msg import Object
from object_detection_msgs.srv import ObjectQuery, ObjectQueryResponse

class ObjectDBService:
    """
    Service for querying object locations

    Usage:
        service = ObjectDBService()
        rospy.spin()
    """
    def __init__(self):
        # Name this node
        rospy.init_node('objectDBService')

        # Params
        self.server = rospy.get_param("~server", "localhost")
        self.port   = rospy.get_param("~port", "6379")
        self.prefix = rospy.get_param("~prefix", "object")

        self.redis = redis.StrictRedis(host=self.server, port=self.port, db=0)

        # Listen to object locations that are published
        rospy.Service("/query_objects", ObjectQuery, self.callback_object)

    def callback_object(self, req):
        """
        Respond to the request for an object's location
        """
        results = []
        data = self.redis.get(self.prefix+"_"+req.name)

        if data:
            data = json.loads(data.decode("utf-8"))

            for r in data:
                o = Object()
                o.name = r["name"]
                o.time = r["time"]
                o.x = r["x"]
                o.y = r["y"]
                o.z = r["z"]
                results.append(o)
        else:
            rospy.logerr("Cannot query database")

        return ObjectQueryResponse(results)

if __name__ == '__main__':
    try:
        service = ObjectDBService()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
