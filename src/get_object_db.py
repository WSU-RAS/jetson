#!/usr/bin/env python2
"""
Example of how to get object locations from the database

rosrun ras_jetson get_object_db.py
"""
import rospy
import psycopg2

class GetObjectDB:
    """
    Access object locations

    Usage:
        db = ObjectDB()
    """
    def __init__(self):
        # Name this node
        rospy.init_node('getObjectDB', anonymous=True)

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

    def get(self, name):
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT x,y,z FROM objects WHERE name = %s", (name,))
            return cur.fetchone()
        except:
            rospy.logerr("could not get object location")

        return None

if __name__ == '__main__':
    try:
        node = GetObjectDB()

        # TODO only for testing right now...
        print "Pillbottle:", node.get("pillbottle")

        rospy.spin()
    except rospy.ROSInterruptException:
        pass
