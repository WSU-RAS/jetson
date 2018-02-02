#!/usr/bin/env python3
"""
Object Detector

Subscribes to RGB images. Publishes bounding boxes from object detection.

The two parts:
 * ObjectDetector is for doing the actual object detection with TensorFlow.
 * ObjectDetectorNode is the ROS node that sends new RGB images to the TensorFlow
   object to process and then publishes the results in a message.
"""
import os
import time
import numpy as np
import tensorflow as tf
from collections import deque

 # Visualization for debugging
from PIL import Image
from matplotlib import pyplot as plt

# ROS
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Messages, either from COB or Darknet's
# From: https://github.com/ipa320/cob_perception_common in cob_perception_msgs
from cob_perception_msgs.msg import Detection, DetectionArray, Rect
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox

# Note: must have models/research/ and models/research/slim/ in PYTHONPATH
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def load_image_into_numpy_array(image):
    """
    Helper function from: models/research/object_detection/
    """
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

class ObjectDetector:
    """
    Object Detection with TensorFlow via their models/research/object_detection

    Based on:
    https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb


    Usage:
        with ObjectDetectorTF("path/to/model_dir.pb", "path/to/tf_label_map.pbtxt") as detector:
            boxes, scores, classes = detector.process(newImage)

    Or:
        detector = ObjectDetectorTF("path/to/model_dir.pb", "path/to/tf_label_map.pbtxt")
        detector.open()
        boxes, scores, classes = detector.process(newImage)
        detector.close()
    """
    def __init__(self, graph_path, labels_path, threshold):
        # Threshold for what to count as objects
        self.threshold = threshold

        # Load frozen TensorFlow model into memory
        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(os.path.join(graph_path, "frozen_inference_graph.pb"), 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        
        # Load label map
        label_map = label_map_util.load_labelmap(labels_path)
        numClasses = len(label_map.item) # Use all classes
        categories = label_map_util.convert_label_map_to_categories(label_map,
                        max_num_classes=numClasses, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    def open(self):
        # Session
        self.session = tf.Session(graph=self.detection_graph)

        #
        # Inputs/outputs to network
        #
        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def close(self):
        self.session.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def process(self, image_np):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Run detection
        (boxes, scores, classes, num) = self.session.run(
            [self.detection_boxes, self.detection_scores,
                self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        # Class label needs to be int32 or else publishing gives error
        classes = classes.astype(np.int32)

        return boxes, scores, classes
        
    def show(self, image_np, boxes, classes, scores, debug_image_size=(12,8)):
        """
        For debugging, show the image with the bounding boxes
        """
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True, line_thickness=8)
        plt.figure(figsize=debug_image_size)
        plt.imshow(image_np)
        plt.show()

    def msgCOB(self, image, boxes, scores, classes):
        """
        Create the Object Detector message to publish with ROS

        From:
        https://github.com/cagbal/cob_people_object_detection_tensorflow/blob/master/src/cob_people_object_detection_tensorflow/utils.py
        """
        msg = DetectionArray()
        msg.header = image.header
        scores_above_threshold = np.where(scores > self.threshold)[1]

        for s in scores_above_threshold:
            # Get the properties
            bb = boxes[0,s,:]
            sc = scores[0,s]
            cl = classes[0,s]

            # Create the detection message
            detection = Detection()
            detection.header = image.header
            detection.label = self.category_index[int(cl)]['name']
            detection.id = cl
            detection.score = sc
            detection.detector = 'Tensorflow object detector'
            detection.mask.roi.x = int((image.width-1) * bb[1])
            detection.mask.roi.y = int((image.height-1) * bb[0])
            detection.mask.roi.width = int((image.width-1) * (bb[3]-bb[1]))
            detection.mask.roi.height = int((image.height-1) * (bb[2]-bb[0]))

            msg.detections.append(detection)

        return msg

    def msgDN(self, image, boxes, scores, classes):
        """
        Create the Object Detector message to publish with ROS

        This uses the Darknet BoundingBox[es] messages
        """
        msg = BoundingBoxes()
        msg.header = image.header
        scores_above_threshold = np.where(scores > self.threshold)[1]

        for s in scores_above_threshold:
            # Get the properties
            bb = boxes[0,s,:]
            sc = scores[0,s]
            cl = classes[0,s]

            # Create the bounding box message
            detection = BoundingBox()
            detection.Class = self.category_index[int(cl)]['name']
            detection.probability = sc
            detection.xmin = int((image.width-1) * bb[1])
            detection.ymin = int((image.height-1) * bb[0])
            detection.xmax = int((image.width-1) * bb[3])
            detection.ymax = int((image.height-1) * bb[2])

            msg.boundingBoxes.append(detection)

        return msg

class ObjectDetectorNode:
    """
    Subscribe to the images and publish object detection results with ROS

    See both of these:
    https://github.com/cagbal/cob_people_object_detection_tensorflow
    https://github.com/tue-robotics/image_recognition/blob/master/tensorflow_ros/scripts/object_recognition_node

    Alternate, uses different message format:
    https://github.com/osrf/tensorflow_object_detector

    Usage:
    with ObjectDetectorNode() as node:
        rospy.spin()
    """
    def __init__(self, averageFPS=60):
        # We'll publish the results
        #self.pub = rospy.Publisher('object_detector', DetectionArray, queue_size=10)
        self.pub = rospy.Publisher('object_detector', BoundingBoxes, queue_size=10)

        # Name this node
        rospy.init_node('object_detector', anonymous=True)

        # Parameters
        graph_path = os.path.expanduser(rospy.get_param("~graph_path"))
        labels_path = os.path.expanduser(rospy.get_param("~labels_path"))
        threshold = rospy.get_param("~threshold", 0.5)
        camera_namespace = rospy.get_param("~camera_namespace", "/camera/rgb/image_rect_color")

        # Object Detector
        self.detector = ObjectDetector(graph_path, labels_path, threshold)
        self.detector.open()

        # For processing images
        self.bridge = CvBridge()

        # For computing average FPS over so many frames
        self.fps = deque(maxlen=averageFPS)

        # Only create the subscriber after we're done loading everything
        self.sub = rospy.Subscriber(camera_namespace, Image, self.rgb_callback, queue_size=1, buff_size=2**24)

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        self.detector.close()

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
            boxes, scores, classes = self.detector.process(image_np)
            #self.pub.publish(self.detector.msgCOB(data, boxes, scores, classes))
            self.pub.publish(self.detector.msgDN(data, boxes, scores, classes))
        except CvBridgeError as e:
            rospy.logerr(e)
            error = "(error)"

        # Print FPS
        fps = 1/(time.time() - fps)
        self.fps.append(fps)
        print("Object Detection FPS", "{:<5}".format("%.2f"%fps),
                "Average", "{:<5}".format("%.2f"%self.avgFPS()),
                error)

def findFiles(folder, prefix="rgb", extension=".png"):
    """
    Find all files recursively in specified folder with a particular
    prefix and extension

    (for running on set of test images)
    """
    files = []
    
    for dirname, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            if filename.startswith(prefix) and filename.endswith(extension):
                files += [(dirname, filename)]
            
    return files

def test(model, root="../networks"):
    """
    Run the object detection on a test set of images
    """
    # Model
    graph = os.path.join(root, model)
    labels = os.path.join(root, "tf_label_map.pbtxt")

    # Test images
    testDir = os.path.join(root, "test_images")
    testImages = [os.path.join(d, f) for d, f in findFiles(testDir)]
    images = []

    # We need to get this out of the inner loop since it's really slow
    for img in testImages:
        images.append(load_image_into_numpy_array(Image.open(img)))

    with ObjectDetector(graph, labels) as detector:
        fps = time.time()

        for img in images: 
            boxes, scores, classes, number = detector.process(img)
            detector.show(img, boxes, scores, classes)

        # Calculate overall FPS
        fps = len(testImages)/(time.time() - fps)
        print("Note, first frame is always really slow...")
        print("Overall FPS", fps)

if __name__ == '__main__':
    try:
        with ObjectDetectorNode() as node:
            rospy.spin()
    except rospy.ROSInterruptException:
        pass

    #test("ssd_mobilenet_v1.pb")
    #test("ssd_inception_v2.pb")
