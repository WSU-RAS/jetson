#!/usr/bin/env python2
import rospy
import psycopg2
from ras_jetson.msg import Object

class UpdateObjectDBNode:
    """
    Save object locations

    Usage:
        node = UpdateObjectDBNode()
        rospy.spin()
    """
    def __init__(self):
        # Name this node
        rospy.init_node('updateObjectDB', anonymous=True)

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

        cur = self.conn.cursor()
        try:
            cur.execute("""CREATE TABLE IF NOT EXISTS objects (
                name varchar PRIMARY KEY,
                x double precision,
                y double precision,
                z double precision);""")
            self.conn.commit()
        except:
            rospy.logfatal("Could not create table")

        # Listen to object locations that are published
        rospy.Subscriber("/find_objects", Object, self.callback_object)

    def callback_object(self, data):
        """
        Save the object location when we see it
        """
        try:
            cur = self.conn.cursor()
            cur.execute("""INSERT INTO objects (name, x, y, z)
                            VALUES (%s, %s, %s, %s)
                        ON CONFLICT (name) DO UPDATE SET
                            x = EXCLUDED.x, y = EXCLUDED.y, z = EXCLUDED.z;
                        """, (data.name, data.x, data.y, data.z))
            self.conn.commit()
        except:
            rospy.logerr("Cannot insert row")

if __name__ == '__main__':
    try:
        node = UpdateObjectDBNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
