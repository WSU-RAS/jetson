#!/usr/bin/env python2
"""
This node listens to bounding boxes from a variety of object detectors and then
uses the camera depth point cloud to calculate the 3D positions relative to the
map (or whatever you set target as).

Subscribes to point cloud from the camera depth sensor:
    /camera/depth_registered/points - Depth data from camera

Subscribes to bounding boxes:
    /darknet_ros/bounding_boxes - YOLO object detection
    /object_detector - TensorFlow object detection
    /human_detector - OpenCV human detection

Publishes to:
    /find_objects - ras_jetson.msg.Object array of 3D positions of objects for
                    the given bounding boxes
    /find_objects_debug - PointStamped x, y, z of last object found
"""
import rospy
import copy
import numpy as np
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped
import sensor_msgs.point_cloud2 as pc2
from darknet_ros_msgs.msg import BoundingBoxes
from ras_jetson_msgs.msg import Object

# Coordinate transformation
import tf2_ros
import PyKDL
from tf2_sensor_msgs.tf2_sensor_msgs import transform_to_kdl

class FindObjectsNode:
    """
    Take the point clouds from the depth sensor and the bounding boxes from the
    object detection to find 3D locations of objects in view

    Usage:
        node = FindObjectsNode()
        rospy.spin()
    """
    def __init__(self):
        # Save cloud in one callback and use it when we receive a bounding box
        self.lastCloud = None

        # We'll publish the results
        self.pub = rospy.Publisher('find_objects', Object, queue_size=30)

        # Name this node
        rospy.init_node('findObjects')

        # Params
        self.target = rospy.get_param("~target", "map")
        self.source = rospy.get_param("~source", "camera_depth_optical_frame")
        self.debug = rospy.get_param("~debug", False)

        # For debugging also publish a point that we can vizualize in rviz
        if self.debug:
            self.debugPoint = rospy.Publisher('find_objects_debug',
                    PointStamped, queue_size=2)

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

            # Also listen for the human detection bounding boxes, hard coded
            # "human" object
            rospy.Subscriber("/human_detector", BoundingBoxes,
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
        Save the newest point cloud
        """
        #self.lastCloud = copy.copy(cloud)
        self.lastCloud = cloud

    def callback_box(self, data):
        """
        Figure out 3D location of objects for which we receive bounding boxes
        """
        if self.lastCloud:
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

                # Can't do anything without the transform
                return

            for b in data.boundingBoxes:
                # Get list of points we want, all of them in the bounding box
                #
                # Note: (u,v) is in image, (x,y,z) is in 3D space
                uvs = []

                for x in range(b.xmin, b.xmax+1):
                    for y in range(b.ymin, b.ymax+1):
                        uvs.append((x,y))

                # Get the points from the point cloud, but ignore NaNs
                points = []
                dist2 = []

                for p in pc2.read_points(self.lastCloud, field_names=("x","y","z"),
                        uvs=uvs, skip_nans=True):
                    points.append((p[0],p[1],p[2]))
                    # distance squared, i.e. sqrt(x^2+y^2+z^2)^2 = x^2+y^2+z^2,
                    # since we only care of min and sqrt doesn't affect this
                    # but is expensive to compute
                    dist2.append(p[0]**2 + p[1]**2 + p[2]**2)

                if len(points) > 0:
                    # Minimum distance should be the best approximation of
                    # where an object is from the depth sensor. Averaging will
                    # yield a point farther back since there will be lots of
                    # other points behind the object in the bounding box.
                    # However, very rarely in our setup will we ever have a
                    # point closer within the bounding box than the object's
                    # location.
                    minindex = np.argmin(np.array(dist2))
                    loc = points[minindex]

                    # Coordinate transformation
                    p = transform_kdl*PyKDL.Vector(loc[0],loc[1],loc[2])

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

if __name__ == '__main__':
    try:
        node = FindObjectsNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
