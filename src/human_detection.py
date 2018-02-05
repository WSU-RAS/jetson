#!/usr/bin/env python2
"""
Human Detector

A test version of the human detector code from Shivam, trying to integrate it into ROS.
https://github.com/WSU-RAS/human-detection/blob/master/facedetector.py
"""
import os
import cv2
import time
import argparse
import numpy as np
from imutils.object_detection import non_max_suppression
from collections import deque

# ROS
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox

def msg(image, boxes):
    """
    Create the Darknet BoundingBox[es] messages
    """
    msg = BoundingBoxes()
    msg.header = image.header

    for (x, y, w, h) in boxes: 
        detection = BoundingBox()
        detection.Class = "human"
        detection.probability = 1 # Not really... but we won't use it anyway
        detection.xmin = x
        detection.ymin = y
        detection.xmax = x+w
        detection.ymax = y+h
        msg.boundingBoxes.append(detection)

    return msg

class HumanDetectorNode:
    """
    Subscribe to the images and publish human detectionresults with ROS

    Usage:
    with ObjectDetectorNode() as node:
        rospy.spin()
    """
    def __init__(self, averageFPS=60):
        # We'll publish the results
        self.pub = rospy.Publisher('human_detector', BoundingBoxes, queue_size=10)

        # Name this node
        rospy.init_node('human_detector', anonymous=True)

        # Parameters
        camera_namespace = rospy.get_param("~camera_namespace", "/camera/rgb/image_rect_color")
	# Not sure if there are any other parameters we really want, e.g.
        #threshold = rospy.get_param("~scale", 1.1)

        # For processing images
        self.bridge = CvBridge()

        # For computing average FPS over so many frames
        self.fps = deque(maxlen=averageFPS)

        # Only create the subscriber after we're done loading everything
        self.sub = rospy.Subscriber(camera_namespace, Image, self.rgb_callback, queue_size=1, buff_size=2**24)

        # initialize the HOG constructor
        self.hog = cv2.HOGDescriptor()

        # use the default pre trained people detector algorithm for HOG + SVM
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def avgFPS(self):
        """
        Return average FPS over last so many frames (specified in constructor)
        """
        return sum(list(self.fps))/len(self.fps)

    def rgb_callback(self, data):
        #print("Object Detection frame at %s" % rospy.get_time())
        fps = time.time()
        error = ""

        try:
            image_np = self.bridge.imgmsg_to_cv2(data, "bgr8")
            boxes = self.processImage(image_np)
            self.pub.publish(msgDN(data, boxes))
        except CvBridgeError as e:
            rospy.logerr(e)
            error = "(error)"

        # Print FPS
        fps = 1/(time.time() - fps)
        self.fps.append(fps)
        print("Object Detection FPS", "{:<5}".format("%.2f"%fps),
                "Average", "{:<5}".format("%.2f"%self.avgFPS()),
                error)

    def processImage(self, frame):
	# convert the RGB image to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# detect people in each frame
	rects, weights = self.hog.detectMultiScale(gray, winStride = (4, 4), padding  = (16, 16), scale = 1.1)

	# apply non maxima supression 
	# to make one bounding box over each human
	# diminish the effect of overlapping bounding boxes

	rects = np.array([[x, y, x+w, y+h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs = None, overlapThresh = 0.65)

	# create bounding boxes over the detected humans
	for (x, y, w, h) in pick:
	    cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
	    coordinates = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]

        print "No of people detected {}".format(len(pick))

        return pick

if __name__ == '__main__':
    try:
        node = HumanDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
