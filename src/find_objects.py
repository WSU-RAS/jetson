#!/usr/bin/env python2
import rospy
import copy
import numpy as np
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped
import sensor_msgs.point_cloud2 as pc2
from darknet_ros_msgs.msg import BoundingBoxes
from cob_perception_msgs.msg import Object

# Coordinate transformation
import tf2_ros
import PyKDL
from tf2_sensor_msgs.tf2_sensor_msgs import transform_to_kdl

class FindObjectsNode:
    """
    Take the point clouds from the depth sensor and the bounding boxes from the
    object detection to find 3D locations of objects in view

    lastSeenTimeout is the number of point cloud frames for which to keep
    updating the object's position after receiving the last bounding box.
    After this time, we'll assume the object is no longer in view and wait till
    we receive another bounding box update.

    Usage:
        node = FindObjectsNode()
        rospy.spin()
    """
    def __init__(self, lastSeenTimeout=30):
        # For saving bounding boxes in one callback and using in another
        self.lastSeen = None
        self.lastSeenCount = 0
        self.lastSeenTimeout = lastSeenTimeout

        # We'll publish the results
        self.pub = rospy.Publisher('find_objects', Object, queue_size=30)

        # Name this node
        rospy.init_node('findObjects', anonymous=True)

        # Params
        self.target = rospy.get_param("~target", "map")
        self.source = rospy.get_param("~source", "camera_depth_optical_frame")
        self.debug = rospy.get_param("~debug", False)

        # For debugging also publish a point that we can vizualize in rviz
        if self.debug:
            self.debugPoint = rospy.Publisher('find_objects_debug',
                    PointStamped, queue_size=30)

        # Listen to reference frames, for the coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Wait till we get the map
        if self.waitTillTransform():
            # Listen to point cloud
            rospy.Subscriber("/camera/depth_registered/points", PointCloud2,
                    self.callback_point)

            # Listen to bounding boxes
            #
            # I made both use the Darknet bounding boxes messages, so either
            # will work (or both, as long as they use the same object names)
            rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes,
                    self.callback_box)
            rospy.Subscriber("/object_detector", BoundingBoxes,
                    self.callback_box)
        else:
            rospy.logerr("failed to get transform from tf2")

    def waitTillTransform(self):
        """
        Block until the desired transform from target to source is available

        See: http://wiki.ros.org/tf2/Tutorials/tf2%20and%20time%20%28Python%29
        """
        found = False
        transform = None

        while not rospy.is_shutdown() and transform == None:
            try:
                transform = self.tf_buffer.lookup_transform(self.target,
                        self.source, rospy.Time(), rospy.Duration(1.0))
                found = True
            except:
                continue

        return found

    def callback_point(self, cloud):
        """
        Handle when we get a new point cloud

        We will find the locations of objects in this rather than in callback_box
        since as the robot is moving, tipping, etc. we want to keep updating where
        the object is. This does assume that the object does not move between
        updated bounding boxes, but we can get 20 fps with SSD MobileNet. It likely
        only matters if using YOLO which right now is ~2.5 fps.
        """
        # If we haven't seen a bounding box update for this many point clouds
        # frames, then set lastSeen back to None. This is so we don't keep using
        # old data when the objects go out of frame.
        self.lastSeenCount += 1

        if self.lastSeenCount > self.lastSeenTimeout:
            self.lastSeen = None

        # Use last bounding boxes we've received (points are at much higher
        # rate than boxes)
        if self.lastSeen:
            points = []

            # Convert point cloud to map reference frame
            try:
                # Arguments: target frame, source frame, time
                transform = self.tf_buffer.lookup_transform(self.target,
                        self.source, rospy.Time())

                # See tf2_sensor_msgs.py
                #
                # We will do it per-point so we don't have to transform the entire
                # point cloud, which is very slow.
                transform_kdl = transform_to_kdl(transform)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                rospy.logerr("error looking up tf2 transform")

            for b in self.lastSeen:
                # Get list of points we want, all of them in the bounding box
                #
                # Note: (u,v) is in image, (x,y,z) is in 3D space
                uvs = []

                for x in range(b.xmin, b.xmax+1):
                    for y in range(b.ymin, b.ymax+1):
                        uvs.append((x,y))

                # Get the points from the point cloud, but ignore NaNs
                points = []

                for p in pc2.read_points(cloud, field_names=("x","y","z"),
                        uvs=uvs, skip_nans=True):
                    points.append((p[0],p[1],p[2]))

                if len(points) > 0:
                    # Average point locations
                    mean = np.array(points).mean(axis=0)

                    # Coordinate transformation
                    p = transform_kdl*PyKDL.Vector(mean[0],mean[1],mean[2])

                    # Publish object update
                    msg = Object()
                    msg.name = b.Class
                    msg.x = p[0]
                    msg.y = p[1]
                    msg.z = p[2]
                    self.pub.publish(msg)

                    if self.debug:
                        msg = PointStamped()
                        # Apparently supposed to be the reference frame for when plotted in rviz
                        msg.header.frame_id = self.target
                        msg.point.x = p[0]
                        msg.point.y = p[1]
                        msg.point.z = p[2]
                        self.debugPoint.publish(msg)

                    rospy.logdebug("%s x %f y %f z %f" %(b.Class,p[0],p[1],p[2]))
                else:
                    rospy.loginfo("all points are NaN in bounding box")

    def callback_box(self, data):
        """
        Handle when we get a new bounding box

        All we do is save it so we'll be able to process it next time we get a
        point cloud, via callback_point
        """
        # We just received a frame, so reset this count
        self.lastSeenCount = 0

        for b in data.boundingBoxes:
            rospy.logdebug("Box: xmin : %d ymin : %d xmax : %d ymax : %d prob : %d class : %s" % (
                    b.xmin, b.ymin, b.xmax, b.ymax, b.probability, b.Class
                ))

        # Copy so we can use in the callback_point function
        self.lastSeen = copy.copy(data.boundingBoxes)

if __name__ == '__main__':
    try:
        node = FindObjectsNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
