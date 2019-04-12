#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from scipy.spatial import KDTree
import tf
import cv2
import yaml
import os
import calendar
import time


STATE_COUNT_THRESHOLD = 2
GENERATE_TRAIN_IMGS = False
DISABLE_CLASSIFIER = False
CLASSIFIER_ALWAYS_GREEN = False

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.camera_image = None
        self.lights = []
        self.is_site = None
        
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/image_color', Image, self.image_cb)
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        
        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.is_site = self.config['is_site']
        self.is_simulator = not self.is_site

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier(self.config['is_site'])
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.imageCounter = 1

        rospy.spin()
        
    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):

        self.lights = msg.lights

    def create_training_data(self, state):
        f_name = "sim_tl_{}_{}.jpg".format(calendar.timegm(time.gmtime()), self.light_label(state))
        dir = './data/train/sim'

        if not os.path.exists(dir):
            os.makedirs(dir)

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image)
        cv_image = cv_image[:, :, ::-1]
        cv2.imwrite('{}/{}'.format(dir, f_name), cv_image)

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint
        Args:
            msg (Image): image from car-mounted camera
        """
        
        # Only classify every forth image to reduce latency
        # Feel free to remove this
        if self.imageCounter % 4 != 0:
            self.imageCounter += 1
            return
        
        self.imageCounter = 1
        
        rospy.loginfo("<---------------- Running image_cb() --------------------------->") #status
        
        self.has_image = True
        self.camera_image = msg

        light_wp, state = self.process_traffic_lights()
        
        # Testing with rosbag
        #light_wp = light_wp if state == TrafficLight.RED else -1
        #rospy.loginfo(light_wp)
        #self.upcoming_red_light_pub.publish(Int32(light_wp))

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state

            if GENERATE_TRAIN_IMGS:
                # Store images and state for training data for simulator
                self.create_training_data(state)

            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
            #Debug
            rospy.loginfo("Publishing incoming red light wp index: {}".format(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
            #Debug
            rospy.loginfo("Publishing incoming red light wp index: {}".format(self.last_wp))
        self.state_count += 1
        

    def get_closest_waypoint(self, pose_x, pose_y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose_x, pose_y (Pose): position to match a waypoint to
        Returns:
            int: index of the closest waypoint in self.waypoints
        """
        #TODO implement
        idx = self.waypoint_tree.query([pose_x, pose_y], 1)[1]
        return idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light
        Args:
            light (TrafficLight): light to classify
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if self.is_simulator:
            rospy.loginfo("Sim ground truth state: {}".format(
                self.light_label(light.state)))
        
        #Use the block below for testing without classifier
        if(self.is_simulator and DISABLE_CLASSIFIER):
            rospy.loginfo('Light state: %s', light.state)
            if(CLASSIFIER_ALWAYS_GREEN):
                return TrafficLight.GREEN
            return light.state

        #Only use the block below for real life testing
        if(not self.has_image):
            self.prev_light_loc = None
            return False


        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, 'rgb8')
        classified_state = self.light_classifier.get_classification(cv_image)

        rospy.loginfo("Classified state:       {}".format(
            self.light_label(classified_state)))

        return classified_state

    def light_label(self, state):
        if state == TrafficLight.RED:
            return "RED"
        elif state == TrafficLight.YELLOW:
            return "YELLOW"
        elif state == TrafficLight.GREEN:
            return "GREEN"
        return "UNKNOWN"


    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color
        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        # only classify lights within dist_threshold to reduce latency
        # Feel free to remove this
        dist_threshold = 100
        closest_light = None
        light_wp_idx = None
        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, 
                                                         self.pose.pose.position.y)

            #TODO find the closest visible traffic light (if one exists)
            dist = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                line = stop_line_positions[i]
                cur_line_wp_idx = self.get_closest_waypoint(line[0], line[1])
                d = cur_line_wp_idx - car_wp_idx
                if d >= 0 and d < dist_threshold and d < dist:
                    dist = d
                    closest_light = light
                    light_wp_idx = cur_line_wp_idx
                
        if closest_light:
            state = self.get_light_state(closest_light)
            if state == TrafficLight.RED:
                rospy.loginfo('Light: RED')
            elif state == TrafficLight.GREEN:
                rospy.loginfo('Light: GREEN')
            elif state == TrafficLight.YELLOW:
                rospy.loginfo('Light: YELLOW')
            elif state == TrafficLight.UNKNOWN:
                rospy.loginfo('Light UNKNOWN')
            return light_wp_idx, state
        #self.waypoints = None
        
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')