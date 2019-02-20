#!/usr/bin/env python2
"""
Object Detector

Subscribes to RGB images. Publishes bounding boxes from object detection.

The two parts:
 * ObjectDetector is for doing the actual object detection with TensorFlow.
 * ObjectDetectorNode is the ROS node that sends new RGB images to the TensorFlow
   object to process and then publishes the results in a message.
"""
import os
import cv2
import time
import pathlib
import numpy as np
import tensorflow as tf

from collections import deque
from tensorflow.contrib.lite.python import interpreter as interpreter_wrapper

# Visualization for debugging
import matplotlib
matplotlib.use('TkAgg')

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ROS (but don't require since not needed for offline mode)
try:
    import rospy
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge, CvBridgeError
    from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox
except ImportError:
    class BoundingBoxes:
        def __init__(self):
            self.header = None
            self.boundingBoxes = []

    class BoundingBox:
        def __init__(self):
            self.Class = None
            self.probability = None
            self.xmin = None
            self.xmax = None
            self.ymin = None
            self.ymax = None

def load_image_into_numpy_array(image):
    """
    Helper function from: models/research/object_detection/
    """
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def load_labels(filename):
    """
    Load labels from the label file

    Note: this is not the tf_label_map.pbtxt, instead just one label per line.
    """
    labels = []

    with open(filename, 'r') as f:
        for l in f:
            labels.append(l.strip())

    return labels

def detection_show(image_np, detection_msg, show_image=True, debug_image_size=(12,8)):
    """ For debugging, show the image with the bounding boxes """
    if len(detection_msg.boundingBoxes) == 0:
        return

    if show_image:
        plt.ion()
        fig, ax = plt.subplots(1, figsize=debug_image_size, num=1)

    for r in detection_msg.boundingBoxes:
        if show_image:
            topleft = (r.xmin, r.ymin)
            width = r.xmax - r.xmin
            height = r.ymax - r.ymin

            rect = patches.Rectangle(topleft, width, height, \
                linewidth=1, edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)
            ax.text(r.xmin, r.ymin, r.Class+": %.2f"%r.probability, fontsize=6,
                bbox=dict(facecolor="y", edgecolor="y", alpha=0.5))

    if show_image:
        ax.imshow(image_np)
        fig.canvas.flush_events()
        #plt.pause(0.05)

def low_level_detection_show(image_np, detection_msg, color=[255,0,0], amt=1):
    """ Overwrite portions on input image with red to display via GStreamer rather than
    with matplotlib which is slow """
    for r in detection_msg.boundingBoxes:
        # left edge
        image_np[r.ymin-amt:r.ymax+amt, r.xmin-amt:r.xmin+amt, :] = color
        # right edge
        image_np[r.ymin-amt:r.ymax+amt, r.xmax-amt:r.xmax+amt, :] = color
        # top edge
        image_np[r.ymin-amt:r.ymin+amt, r.xmin-amt:r.xmax+amt, :] = color
        # bottom edge
        image_np[r.ymax-amt:r.ymax+amt, r.xmin-amt:r.xmax+amt, :] = color

def create_detection_msg(image, boxes, scores, classes, labels, min_score):
    """
    Create the Object Detector message to publish with ROS

    This uses the Darknet BoundingBox[es] messages
    """
    msg = BoundingBoxes()
    msg.header = image.header
    scores_above_threshold = np.where(scores > min_score)[1]

    for s in scores_above_threshold:
        # Get the properties
        bb = boxes[0,s,:]
        sc = scores[0,s]
        cl = classes[0,s]

        # Create the bounding box message
        detection = BoundingBox()
        detection.Class = labels[int(cl)]
        detection.probability = sc
        detection.xmin = int((image.width-1) * bb[1])
        detection.ymin = int((image.height-1) * bb[0])
        detection.xmax = int((image.width-1) * bb[3])
        detection.ymax = int((image.height-1) * bb[2])

        msg.boundingBoxes.append(detection)

    return msg

class TFObjectDetector:
    """
    Object Detection with TensorFlow model trained with
    models/research/object_detection (Non-TF Lite version)

    Based on:
    https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

    Usage:
        with TFObjectDetector("path/to/model_dir.pb", "path/to/labels.txt", 0.5)
            detection_msg = d.process(image, image_np)
    """
    def __init__(self, graph_path, labels_path, min_score,
            memory=0.5, width=300, height=300):
        # Prune based on score
        self.min_score = min_score

        # Model dimensions
        self.model_input_height = height
        self.model_input_width = width

        # Max memory usage (0 - 1)
        self.memory = memory

        # Load frozen TensorFlow model into memory
        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(os.path.join(graph_path, "frozen_inference_graph.pb"), 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Load label map -- index starts with 1 for the non-TF Lite version
        self.labels = ["???"] + load_labels(labels_path)

    def open(self):
        # Config options: max GPU memory to use.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.memory)
        config = tf.ConfigProto(gpu_options=gpu_options)

        # Session
        self.session = tf.Session(graph=self.detection_graph, config=config)

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

    def model_input_dims(self):
        """ Get desired model input dimensions """
        return (self.model_input_width, self.model_input_height)

    def close(self):
        self.session.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def process(self, image, image_np):
        # Expand dimensions since the model expects images to have shape:
        #   [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Run detection
        (boxes, scores, classes, num) = self.session.run(
            [self.detection_boxes, self.detection_scores,
                self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        # Make detection message
        return create_detection_msg(image, boxes, scores, classes,
            self.labels, self.min_score)

class TFLiteObjectDetector:
    """
    Object Detection with TensorFlow model trained with
    models/research/object_detection (TF Lite version)

    Based on:
    https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

    Usage:
        d = TFLiteObjectDetector("path/to/model_file.tflite", "path/to/tf_label_map.pbtxt", 0.5)
        detection_msg = d.process(image, image_np)
    """
    def __init__(self, model_file, labels_path, min_score):
        # Prune based on score
        self.min_score = min_score

        # TF Lite model
        self.interpreter = interpreter_wrapper.Interpreter(model_path=model_file)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        if self.input_details[0]['dtype'] == type(np.float32(1.0)):
            self.floating_model = True
        else:
            self.floating_model = False

        # NxHxWxC, H:1, W:2
        self.model_input_height = self.input_details[0]['shape'][1]
        self.model_input_width = self.input_details[0]['shape'][2]

        # Load label map
        self.labels = load_labels(labels_path)

    def model_input_dims(self):
        """ Get desired model input dimensions """
        return (self.model_input_width, self.model_input_height)

    def process(self, image, image_np, input_mean=127.5, input_std=127.5):
        # Normalize if floating point (but not if quantized)
        if self.floating_model:
            image_np = (np.float32(image_np) - input_mean) / input_std

        # Expand dimensions since the model expects images to have shape:
        #   [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Pass image to the network
        self.interpreter.set_tensor(self.input_details[0]['index'], image_np_expanded)

        # Run
        self.interpreter.invoke()

        # Get results
        detection_boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
        detection_classes = self.interpreter.get_tensor(self.output_details[1]['index'])
        detection_scores = self.interpreter.get_tensor(self.output_details[2]['index'])
        num_detections = self.interpreter.get_tensor(self.output_details[3]['index'])

        num_detections = self.interpreter.get_tensor(self.output_details[3]['index'])

        if not self.floating_model:
            box_scale, box_mean = self.output_details[0]['quantization']
            class_scale, class_mean = self.output_details[1]['quantization']

            # If these are zero, then we end up setting all our results to zero
            if box_scale != 0:
                detection_boxes = (detection_boxes - box_mean * 1.0) * box_scale
            if class_mean != 0:
                detection_classes = (detection_classes - class_mean * 1.0) * class_scale

        # Make detection message
        return create_detection_msg(image,
            detection_boxes, detection_scores, detection_classes,
            self.labels, self.min_score)

class ObjectDetectorBase(object):
    """ Wrap detector to calculate FPS """
    def __init__(self, model_file, labels_path, min_score=0.5, memory=0.5,
            average_fps_frames=30, debug=True, lite=True):
        self.debug = debug
        self.lite = lite

        if lite:
            self.detector = TFLiteObjectDetector(model_file, labels_path,
                min_score)
        else:
            self.detector = TFObjectDetector(model_file, labels_path,
                min_score, memory)

        # compute average FPS over # of frames
        self.fps = deque(maxlen=average_fps_frames)

        # compute streaming FPS (how fast frames are arriving from camera
        # and we're able to process them, i.e. this is the actual FPS)
        self.stream_fps = deque(maxlen=average_fps_frames)
        self.process_end_last = 0

    def open(self):
        if not self.lite:
            self.detector.open()

    def __enter__(self):
        self.open()
        return self

    def close(self):
        if not self.lite:
            self.detector.close()

    def __exit__(self, type, value, traceback):
        self.close()

    def avg_fps(self):
        """ Return average FPS over last so many frames (specified in constructor) """
        return sum(list(self.fps))/len(self.fps)

    def avg_stream_fps(self):
        """ Return average streaming FPS over last so many frames (specified in constructor) """
        return sum(list(self.stream_fps))/len(self.stream_fps)

    def process(self, *args, **kwargs):
        if self.debug:
            # Start timer
            fps = time.time()

        detections = self.detector.process(*args, **kwargs)

        if self.debug:
            now = time.time()

            # End timer
            fps = 1/(now - fps)
            self.fps.append(fps)

            # Streaming FPS
            stream_fps = 1/(now - self.process_end_last)
            self.stream_fps.append(stream_fps)
            self.process_end_last = now

            print "Object Detection", \
                "Process FPS", "{:<5}".format("%.2f"%self.avg_fps()), \
                "Stream FPS", "{:<5}".format("%.2f"%self.avg_stream_fps())

        return detections

class ObjectDetectorNode(ObjectDetectorBase):
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
    def __init__(self):
        # We'll publish the results
        self.pub = rospy.Publisher('object_detector', BoundingBoxes, queue_size=10)

        # Name this node
        rospy.init_node('object_detector')

        # Parameters
        graph_path = os.path.expanduser(rospy.get_param("~graph_path"))
        labels_path = os.path.expanduser(rospy.get_param("~labels_path"))
        threshold = rospy.get_param("~threshold", 0.5)
        memory = rospy.get_param("~memory", 0.5) # 0 to 1
        self.debugImage = rospy.get_param("~debugImage", False)
        camera_namespace = rospy.get_param("~camera_namespace", "/camera/rgb/image_rect_color")

        # Debug images
        if self.debugImage:
            self.pubImage = rospy.Publisher(
                    "/detection_image", Image, queue_size=1)

        # For processing images
        self.bridge = CvBridge()

        # Only create the subscriber after we're done loading everything
        self.sub = rospy.Subscriber(camera_namespace, Image, self.rgb_callback, queue_size=1, buff_size=2**24)

        # Initialize object detector -- the base class with arguments from ROS
        super(ObjectDetectorNode, self).__init__(graph_path, labels_path, threshold, memory)

    def image_msg(self, image_np, detection_msg):
        """
        Create the debug image with bounding boxes on it
        """
        image_np = low_level_detection_show(image_np, detection_msg)
        return self.bridge.cv2_to_imgmsg(image_np, encoding="bgr8")

    def rgb_callback(self, image):
        try:
            # TODO do we need a resize here to 300x300?
            image_np = self.bridge.imgmsg_to_cv2(image, "bgr8")
            detection_msg = self.detector.process(image, image_np)
            self.pub.publish(detection_msg)

            if self.debugImage:
                img_msg = self.image_msg(image, image_np, detection_msg)
                self.pubImage.publish(img_msg)
        except CvBridgeError as e:
            rospy.logerr(e)

class DummyImageMsg:
    """ Allow for using same functions as in the ROS code to get width/height
    and header from some ROS image message """
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.header = ""

class OfflineObjectDetector(ObjectDetectorBase):
    """ Run object detection on already captured images """
    def __init__(self, *args, **kwargs):
        super(OfflineObjectDetector, self).__init__(*args, **kwargs)

    def run(self, test_image_dir, show_image=True):
        test_images = [str(f) for f in pathlib.Path(test_image_dir).glob("*")]

        try:
            for i, filename in enumerate(test_images):
                orig_img = Image.open(filename)

                if orig_img.size == self.detector.model_input_dims():
                    orig_img = load_image_into_numpy_array(orig_img)
                    resize_img = orig_img
                else:
                    resize_img = orig_img.resize(self.detector.model_input_dims())
                    orig_img = load_image_into_numpy_array(orig_img)
                    resize_img = load_image_into_numpy_array(resize_img)

                img = DummyImageMsg(orig_img.shape[1], orig_img.shape[0])
                detection_msg = self.process(img, resize_img)

                if self.debug:
                    for i, d in enumerate(detection_msg.boundingBoxes):
                        print "Result "+str(i)+":", d.Class, d.probability, d.xmin, d.xmax, d.ymin, d.ymax

                detection_show(orig_img, detection_msg, show_image)
        except KeyboardInterrupt:
            pass

if __name__ == '__main__':
    offline = False

    if offline:
        #model = "networks/ssd_mobilenet_v1_laptop.pb"
        model = "networks/ssd_mobilenet_v1_laptop.tflite"
        labels = "networks/labels_laptop.txt"
        with OfflineObjectDetector(model, labels, lite=True) as d:
            d.run("test_images", show_image=True)
    else:
        try:
            with ObjectDetectorNode() as node:
                rospy.spin()
        except rospy.ROSInterruptException:
            pass
