#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree
#
import numpy as np
from tf.transformations import (
    quaternion_matrix, 
    quaternion_from_matrix, 
    # quaternion_from_euler, 
    # euler_from_quaternion, 
    # quaternion_multiply
)


STATE_COUNT_THRESHOLD = 3

# Data collection
#--------------------------------#
is_collecting_traffic_data = True
# is_collecting_traffic_data = False
data_dir_str = "/capstone/traffic_light_data/"
file_prefix = "tl"
tl_data_count = 0
#--------------------------------#

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        # Variables
        self.base_waypoints = None
        self.waypoints_2d   = None
        self.waypoint_tree  = None

        # Data collection
        self.tl_data_count = 0
        # Camera intrinsic matrix (Ground truth)
        f_camera = 1345.0
        # f_camera = 100
        #
        fx_camera = f_camera
        fy_camera = f_camera
        xo_camera = 800/2.0
        yo_camera = 600/2.0
        self.np_K_camera_est = np.array([[fx_camera, 0.0, xo_camera], [0.0, fy_camera, yo_camera], [0.0, 0.0, 1.0]]) # Estimated
        print("np_K_camera_est = \n%s" % str(self.np_K_camera_est))
        #
        self.R_car_at_camera = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)
            print("len(self.waypoints_2d) = %d" % len(self.waypoints_2d))

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

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
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        closest_idx = self.waypoint_tree.query([x,y], 1)[1]
        return closest_idx

    def get_relative_pose(self, pose_obj, pose_ref):
        '''
        Calculate the transformation between the object and the reference frame
        Specifically, the object pose represents in reference frame.
        '''
        point_obj = np.array( (pose_obj.position.x, pose_obj.position.y, pose_obj.position.z) ).reshape((3,1))
        point_ref = np.array( (pose_ref.position.x, pose_ref.position.y, pose_ref.position.z) ).reshape((3,1))
        q_obj = (pose_obj.orientation.x, pose_obj.orientation.y, pose_obj.orientation.z, pose_obj.orientation.w)
        q_ref = (pose_ref.orientation.x, pose_ref.orientation.y, pose_ref.orientation.z, pose_ref.orientation.w)
        #
        T_obj = quaternion_matrix(q_obj) # 4x4 matrix
        T_ref = quaternion_matrix(q_ref)
        # T_id = quaternion_matrix([0.0, 0.0, 0.0, 1.0])
        # print("T_obj = \n%s" % T_obj)
        # print("T_ref = \n%s" % T_ref)
        # print("R_id = \n%s" % R_id)
        #
        T_obj[0:3,3:] = point_obj
        T_ref[0:3,3:] = point_ref
        # T_rel = T_wold_at_ref * T_obj_at_world --> T_rel = T_ref^-1 * T_obj
        T_rel = (np.linalg.inv(T_ref)).dot(T_obj)

        print("T_obj = \n%s" % T_obj)
        print("T_ref = \n%s" % T_ref)
        print("T_rel = \n%s" % T_rel)

        R_rel = T_rel[0:3,0:3]
        t_rel = T_rel[0:3,3:4]
        # q_rel = quaternion_from_matrix(R_rel)
        return (T_rel, R_rel, t_rel)

    def perspective_projection(self, R_tl_at_car, t_tl_at_car, point_at_tl_list):
        '''
        This function help project the points represented in traffic light local frame onto the image
        '''
        _R_tl_at_camera = self.R_car_at_camera.dot(R_tl_at_car)
        _t_tl_at_camera = self.R_car_at_camera.dot(t_tl_at_car)
        projection_list = list()
        for _p_3D_at_tl in point_at_tl_list:
            #
            _point_3D_at_tl = np.array(_p_3D_at_tl).reshape((3,1))
            _point_3D_at_camera = _R_tl_at_camera.dot(_point_3D_at_tl) + _t_tl_at_camera
            _ray = self.np_K_camera_est.dot( _point_3D_at_camera )
            _projection = (_ray / abs(_ray[2,0]))[:2,0]
            print("_projection = \n%s" % _projection)
            projection_list.append(_projection)
        return projection_list



    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        '''
        Since the process is quit similar for collecting traffic light data 
        (i.e. find the closest light, its location, and its state), 
        I reuse the code for data collecting.
        '''
        if not is_collecting_traffic_data:
            # For testing, simply return the light state (for simulation only)
            return light.state

            # # The Following codes are for classification
            # if(not self.has_image):
            #     self.prev_light_loc = None
            #     return False

            # cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

            # #Get classification
            # return self.light_classifier.get_classification(cv_image)
        else:
            # Collect traffic image, location (bounding box), and state
            if(not self.has_image):
                self.prev_light_loc = None
                return False
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
            # Count the image
            self.tl_data_count += 1

            print("--- tl_data_count = %d ---" % self.tl_data_count)
            # Try to get the bounding box
            print("light.pose = \n%s" % light.pose)
            print("self.pose = \n%s" % self.pose)


            # Perspective projection
            #---------------------------------#
            # Calculate the relative pose of light at car
            T_rel, R_tl_at_car, t_tl_at_car = self.get_relative_pose(light.pose.pose, self.pose.pose)
            print("R_tl_at_car = \n%s" % R_tl_at_car)
            print("t_tl_at_car = \n%s" % t_tl_at_car)
            #
            # _light_center_point_at_camera = self.R_car_at_camera.dot(t_tl_at_car)
            # print("_light_center_point_at_camera = \n%s" % _light_center_point_at_camera)
            #
            # _ray = self.np_K_camera_est.dot( _light_center_point_at_camera)
            # _projection = (_ray / abs(_ray[2,0]))[:2,0]
            # print("_projection = \n%s" % _projection)
            point_at_tl_list = list()
            point_at_tl_list.append([0., 0., 0.])
            projection_list = self.perspective_projection(R_tl_at_car, t_tl_at_car, point_at_tl_list)
            #---------------------------------#

            # TODO: Generate the bounding box
            # TODO: TRy drawing the boundinf box on the image

            # Store the image
            _file_name = file_prefix + ("_%.4d_%d" % (self.tl_data_count, light.state)) + ".png"
            data_path_str = data_dir_str + _file_name
            cv2.imwrite(data_path_str, cv_image )
            #
            return light.state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

            #TODO find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                # Get stop line waypoint index
                line = stop_line_positions[i] # Note: this is loaded from config
                tmp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                # Find closest stop line waypoint index
                d = tmp_wp_idx - car_wp_idx
                if d >= 0 and d < diff:
                    # Found a closer frontal light (stop line)
                    diff = d
                    closest_light = light
                    line_wp_idx = tmp_wp_idx


        if closest_light:
            state = self.get_light_state(closest_light)
            return line_wp_idx, state
        # self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
