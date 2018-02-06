#!/usr/bin/env python2
import rospy
import psycopg2
from ras_jetson.msg import Object
from ras_jetson.srv import ObjectQuery, ObjectQueryResponse

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
        self.dbname = rospy.get_param("~db", "ras")
        self.server = rospy.get_param("~server", "localhost")
        self.username = rospy.get_param("~user", "ras")
        self.password = rospy.get_param("~pass", "ras")

        try:
            self.conn = psycopg2.connect(
                    "dbname='%s' user='%s' host='localhost' password='%s'"%(
                        self.dbname, self.username, self.password))
        except:
            rospy.logfatal("unable to connect to database")

        # Listen to object locations that are published
        rospy.Service("/query_objects", ObjectQuery, self.callback_object)

    def callback_object(self, req):
        """
        Respond to the request for an object's location
        """
        results = []

        try:
            cur = self.conn.cursor()
            cur.execute("""SELECT name, x, y, z FROM objects WHERE name = %s""",
                    (req.name,))
            self.conn.commit()
            rows = cur.fetchall()

            for r in rows:
                o = Object()
                o.name = r[0]
                o.x = r[1]
                o.y = r[2]
                o.z = r[3]
                results.append(o)
        except:
            rospy.logerr("Cannot query database")

        return ObjectQueryResponse(results)

if __name__ == '__main__':
    try:
        service = ObjectDBService()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
