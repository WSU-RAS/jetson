#!/usr/bin/env python2
"""
Provide a service for querying object locations saved in Redis DB. Also
provides a way to load fake locations from a file except for the human
for when running experiments.

Access via:
    /query_objects

See the get_object_service.py example code for how to use this. Or, to test,
you can use:
    rosservice call /query_objects pillbottle
"""
import json
import yaml
import rospy
import redis
import datetime
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

        # Load some locations from a .yaml file rather than live if desired
        filename = rospy.get_param("~file", "")
        self.locations = None

        if len(filename) > 0:
            with open(filename) as f:
                self.locations = yaml.load(f)

        self.redis = redis.StrictRedis(host=self.server, port=self.port, db=0)

        # Listen to object locations that are published
        rospy.Service("/query_objects", ObjectQuery, self.callback_object)

    def callback_object(self, req):
        """
        Respond to the request for an object's location
        """
        results = []
        fake = False

        # Load from the file if specified and if it's in the file
        #
        # Note: this means we want to not specify "human" in the file since
        # we want that to be live
        if self.locations != None and req.name in self.locations:
            # Note: when loading from a file we have saved the z/w for
            # rotation, but actually we'd want to calculate that rather than
            # storing in the database. The database is the locations of the
            # objects not the locations/rotations for the robot when navigating
            # to the object.
            data = [{
                    "name": req.name,
                    "time": datetime.datetime.utcnow().isoformat(),
                    "x": self.locations[req.name]["x"],
                    "y": self.locations[req.name]["y"],
                    "z": self.locations[req.name]["z"],
                    "w": self.locations[req.name]["w"]
                    }]
            fake = True
        else:
            data = self.redis.get(self.prefix+"_"+req.name)

            if data:
                data = json.loads(data.decode("utf-8"))
            else:
                rospy.logerr("Cannot query database")

        for r in data:
            o = Object()
            o.name = r["name"]
            o.time = r["time"]
            o.x = r["x"]
            o.y = r["y"]

            if fake:
                o.z = r["z"]
            else:
                o.z = 0

            # We actually aren't saving w's in the database but do have it when
            # loading from a file
            if "w" in r:
                o.w = r["w"]
            else:
                o.w = 1

            results.append(o)

        return ObjectQueryResponse(results)

if __name__ == '__main__':
    try:
        service = ObjectDBService()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
