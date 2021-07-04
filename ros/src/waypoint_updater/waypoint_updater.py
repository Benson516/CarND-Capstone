#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
import numpy as np

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 0.5 # m/s^2

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

         # TODO: Add other member variables you need below
        self.base_lane = None
        self.pose = None
        self.stopline_wp_idx = -1
        self.waypoints_2d   = None
        self.waypoint_tree  = None


        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)


        # rospy.spin()
        self.loop()

    def loop(self):
        """
        """
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.pose and self.base_lane:
                if self.waypoints_2d and self.waypoint_tree:
                    self.publish_waypoints()
            rate.sleep()

    def get_closest_waypoint_idx(self):
        """
        """
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        # Get "1" closest point's index. Note: return of the query() is (point, idx)
        closest_idx = self.waypoint_tree.query([x,y], 1)[1]

        # Check if the closest is ahead or behind the vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]

        # Equation for the hyperplane through closest_coord
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pose_vect = np.array([x,y])

        val = np.dot(cl_vect-prev_vect, pose_vect-cl_vect)

        if val > 0:
            # The closest is behind the vehicle
            # print("The closest waypoint is behind the vehicle, use the next point.")
            closest_idx = (closest_idx+1) % len(self.waypoints_2d)
        #
        print("closest_idx = %d" % closest_idx)
        return closest_idx

    def publish_waypoints(self):
        """
        """
        # # 1st version
        # lane = Lane()
        # lane.header = self.base_lane.header
        # lane.waypoints = self.base_lane.waypoints[closest_idx:(closest_idx + LOOKAHEAD_WPS)]
        # self.final_waypoints_pub.publish(lane)

        # 2nd version
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self):
        """
        """
        lane = Lane()

        closest_idx = self.get_closest_waypoint_idx()
        farest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_lane.waypoints[closest_idx:farest_idx]
        if farest_idx >= len(self.base_lane.waypoints):
            base_waypoints += self.base_lane.waypoints[:(farest_idx-len(self.base_lane.waypoints)+1)]

        if (self.stopline_wp_idx == -1) or (self.stopline_wp_idx >= farest_idx):
            lane.waypoints = base_waypoints
        else:
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)

        return lane

    def decelerate_waypoints(self, waypoints, closest_idx):
        """
        """
        temp = []
        for i, wp in enumerate(waypoints):

            p = Waypoint()
            p.pose = wp.pose

            stop_idx = max(self.stopline_wp_idx - closest_idx - 2, 0) # Two waypoints back from line so the front of the car stops at line 
            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt( 2 * MAX_DECEL * dist)
            if vel < 1.0:
                vel = 0.0

            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)
        
        return temp

    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg
        # print("self.pose = %s" % str(self.pose))

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        self.base_lane = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)
            print("len(self.waypoints_2d) = %d" % len(self.waypoints_2d))

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        _length = len(waypoints)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1%_length].pose.pose.position, waypoints[i%_length].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
