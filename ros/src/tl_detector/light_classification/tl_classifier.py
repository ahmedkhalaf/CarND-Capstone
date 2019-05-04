from styx_msgs.msg import TrafficLight
import cv2
import os
import tensorflow as tf
import numpy as np
import rospy

class TLClassifier(object):
    def __init__(self, is_site):
        #TODO load classifier
        self.light = TrafficLight.UNKNOWN
        self.category_index = {1: {'id': 1, 'name': 'GREEN'}, 2: {'id': 2, 'name': 'RED'},
                               3: {'id': 3, 'name': 'YELLOW'}, 4: {'id': 4, 'name': 'UNKNOWN'}}
        optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1,do_function_inlining=True)
        graph_options = tf.GraphOptions(optimizer_options=optimizer_options)
        config=tf.ConfigProto(log_device_placement=True,graph_options=graph_options)
        config.gpu_options.allow_growth = True
        if is_site:
            PATH_TO_MODEL = 'ssd_inception_v2_carla/frozen_inference_graph.pb'
        else:
            PATH_TO_MODEL = 'ssd_inception_v2_sim/frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name = '')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph = self.detection_graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        #TODO implement light color prediction
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(image_rgb, axis = 0)
        (boxes, scores, classes, num) = self.sess.run(
            [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
            feed_dict = {self.image_tensor: image_expanded})
        #return TrafficLight.RED
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)
        num = np.squeeze(num).astype(np.int32)
        #rospy.loginfo('get_classification: %s, %s, %s, %s', boxes, scores, classes, num)

        min_threshold = 0.3
        count_total = 0
        count_red = count_green = count_yellow = 0
        for i in range(boxes.shape[0]):
            if scores is None or scores[i] > min_threshold:
                count_total += 1

                class_name = self.category_index[classes[i]]['name']
                if class_name == 'RED':
                    count_red += 1

        #s = 'UNKNOWN'
        if count_red < count_total - count_red:
            self.light = TrafficLight.GREEN
            #s = 'GREEN'
        else:
            self.light = TrafficLight.RED
            #s = 'RED'

        #rospy.loginfo('Light: %s', s)
        return self.light
