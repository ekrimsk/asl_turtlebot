#!/usr/bin/env python

import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Twist, Pose2D, PoseStamped
from asl_turtlebot.msg import DetectedObject
from std_msgs.msg import String
import tf
import numpy as np
from numpy import linalg
from utils import wrapToPi
from planners import AStar, compute_smoothed_traj
from grids import StochOccupancyGrid2D
import scipy.interpolate
import matplotlib.pyplot as plt
from controllers import PoseController, TrajectoryTracker, HeadingController
from enum import Enum

from dynamic_reconfigure.server import Server
from asl_turtlebot.cfg import NavigatorConfig

# state machine modes, not all implemented
class VendorMode(Enum):
    HOME = 0
    EXPLORE = 1
    HOMING = 2
    DELIVER = 3

class Expliver:
    """
    This node handles high-level state machine that switches between exploration
    and delivery.
    """
    def __init__(self):
        rospy.init_node('turtlebot_expliver', anonymous=True)
        self.mode = VendorMode.HOME

        # initialize variables
        self.exploration_done = False
        self.vendor_locations = {'apple':None, 'hot_dog':None, 'banana':None}
        self.delivery_requests = [] # updated by TA requests for food
        self.vendor_min_dist = rospy.get_param("~vendor_min_dist",0.7)
        self.x = 0.0   # current state (see callback)
        self.y = 0.0
        self.theta = 0.0
        self.at_thresh = 0.02
        self.at_thresh_theta = 0.05
        self.vendor_stop_time = 8.  # how long we are stopping
        self.vendor_arrival_time = None  # the time we started the vendor stop
        self.vendor_stop = False # not currently doing a vendor stop

        # Subscriptions
        rospy.Subscriber('/detector/apple', DetectedObject, self.vendor_detected_callback)
        rospy.Subscriber('/detector/hot_dog', DetectedObject, self.vendor_detected_callback)
        rospy.Subscriber('/detector/banana', DetectedObject, self.vendor_detected_callback)
        rospy.Subscriber('/current_state',Pose2D, self.current_state_callback)
        rospy.Subscriber('/delivery_request',String,self.delivery_request_callback)

	# Publishers
        # self.vendor_stop_pub = rospy.Publisher('/vendor_stop', String, queue_size=10)
        self.goal_pub = rospy.Publisher('/cmd_nav', Pose2D, queue_size=10)

        # Initialize robot home location
        self.home = None

        print "finished expliver init"

    ####### CALLBACK FUNCTIONS ######

    # update current state by subscribing to '/current_state'
    def current_state_callback(self,msg):
        # save first state reading as home location of robot
        if self.home is None:
            self.home = Pose2D()
            self.home.x = msg.x
            self.home.y = msg.y
            self.home.theta = msg.theta
        # update state
        self.x = msg.x
        self.y = msg.y
        self.theta = msg.theta

    # Add vendor locations to dictionary during EXPLORE mode
    def vendor_detected_callback(self,msg):
        dist = msg.distance # distance of detected object
        vendor = msg.name # name of vendor detected

        if not self.vendor_locations.has_key(vendor):
            pass
	elif self.mode == VendorMode.EXPLORE:
            if dist > 0 and dist < self.vendor_min_dist and self.vendor_locations[vendor] is None:
                # SAVE CURRENT ROBOT LOCATION AS VENDOR LOCATION NEED TO CHANGE
                location = Pose2D()
                location.x = self.x
                location.y = self.y
                location.theta = self.theta
                self.vendor_locations[vendor] = location
                rospy.loginfo("We saw vendor " + vendor + " and stored location")
                # self.vendor_stop_pub.publish(vendor)
                # ADD MARKER TO RVIZ BY PUBLISHING MARKER
                # YOUR CODE HERE


    # add delivery requests to list if not already there
    def delivery_request_callback(self,msg):
        if self.mode == VendorMode.HOME:
            self.switch_mode(VendorMode.DELIVER)
        # WE NEED TO UPDATE FOR MORE THAN ONE ITEM
        # msg will be all like "apple, hot_dog, banana"
        if len(self.delivery_requests) != 0 and self.mode == VendorMode.DELIVER:
            self.delivery_requests.append(msg)

    def shutdown_callback(self):
        self.goal_pub.publish(self.home)

    ##### HELPER FUNCTIONS #####

    def switch_mode(self, new_mode):
        # switch mode with print statement
        rospy.loginfo("EXPLIVER: switching from %s -> %s", self.mode, new_mode)
        self.mode = new_mode

    def at_goal(self,goal):
        # returns True if we are at the goal position
        return (linalg.norm(np.array([self.x-goal.x, self.y-goal.y])) < self.at_thresh and abs(wrapToPi(self.theta - goal.theta)) < self.at_thresh_theta)

    def init_vendor_stop(self):
        vendor = self.delivery_requests[0]
        # print that we're at the vendor
        rospy.loginfo("Reached vendor " + vendor)
        self.vendor_arrival_time = rospy.get_rostime()

    def has_finished_vendor_stop(self):
        return rospy.get_rostime() - self.vendor_arrival_time > rospy.Duration.from_sec(self.vendor_stop_time)

    ##### STATE MACHINE LOOP #####
    def run(self):
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():
        
            if self.mode == VendorMode.HOME:
                if not self.exploration_done:
                    self.switch_mode(VendorMode.EXPLORE)
                else:
                    pass

            elif self.mode == VendorMode.EXPLORE:
                # if we have locations for all vendors, go home
                if all(x is not None for x in self.vendor_locations.values()):
                    self.switch_mode(VendorMode.HOMING)
                    self.exploration_done = True

            elif self.mode == VendorMode.HOMING:
                # set goal location to self.home (Pose2D)
                self.goal_pub.publish(self.home)
                if self.at_goal(self.home):
                    self.switch_mode(VendorMode.HOME)

            elif self.mode == VendorMode.DELIVER:
                # if there are no more vendors, go home
                if len(self.delivery_requests) == 0:
                    self.switch_mode(VendorMode.HOMING)
                else:
                    # set first entry in delivery_requests as the goal location
                    delivery_goal = self.vendor_locations[self.delivery_requests[0]]
                    self.goal_pub.publish(delivery_goal)
                    # check if we are at the vendor and haven't initialized stop
                    if self.at_goal(delivery_goal) and not self.vendor_stop:
                        # start the vendor stop
                        self.init_vendor_stop()
                        self.vendor_stop = True
                    elif self.at_goal(delivery_goal) and self.vendor_stop:
                        if self.has_finished_vendor_stop():
                            self.delivery_requests.pop(0)
                            self.vendor_stop = False
            else:
                raise Exception("Idk wtf is happening")
            rate.sleep()

if __name__ == '__main__':
    expliver = Expliver()
    rospy.on_shutdown(expliver.shutdown_callback)
    expliver.run()
                







