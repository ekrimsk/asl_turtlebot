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
from visualization_msgs.msg import Marker

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
        self.vendor_locations = {'apple':None, 'hot_dog':None, 'banana':None}  # actual vendor and marker locations 
        self.vendor_nav_locations = {'apple':None, 'hot_dog':None, 'banana':None} # where we will tell the robot to drive to 
        self.delivery_requests = [] # updated by TA requests for food
        self.vendor_min_dist = rospy.get_param("~vendor_min_dist",0.7)
        self.x = 0.0   # current state (see callback)
        self.y = 0.0
        self.theta = 0.0
        self.at_thresh = 0.091 # was 0.1
        self.at_thresh_theta = 0.21
        self.vendor_stop_time = 5.  # how long we are stopping
        self.vendor_arrival_time = None  # the time we started the vendor stop
        self.vendor_stop = False # not currently doing a vendor stop

        self.goal = Pose2D() # Pose2D goal
        self.goal.x = 0.0
        self.goal.y = 0.0
        self.goal.theta = 0.0

        # for dog 
        self.dog_seen = False
        # Subscriptions
        rospy.Subscriber('/detector/apple', DetectedObject, self.vendor_detected_callback)
        rospy.Subscriber('/detector/hot_dog', DetectedObject, self.vendor_detected_callback)
        rospy.Subscriber('/detector/banana', DetectedObject, self.vendor_detected_callback)
        rospy.Subscriber('/detector/dog', DetectedObject, self.dog_detected_callback)
        rospy.Subscriber('/current_state',Pose2D, self.current_state_callback)
        rospy.Subscriber('/delivery_request',String,self.delivery_request_callback)
        rospy.Subscriber('/cmd_nav', Pose2D, self.goal_callback)
        # map 
        # rospy.Subscriber('/map', OccupancyGrid, self.map_callback)

	# Publishers
        # self.vendor_stop_pub = rospy.Publisher('/vendor_stop', String, queue_size=10)
        self.goal_pub = rospy.Publisher('/cmd_nav', Pose2D, queue_size=10)
        self.vendor_marker_pub = rospy.Publisher('/vendor_marker', Marker, queue_size=10)
        self.dog_marker_pub = rospy.Publisher('/dog_marker', Marker, queue_size=10)

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
    def dog_detected_callback(self,msg):
        #rospy.loginfo("WooofWoof! A dog!")
        # Hacky fix to reduce distance
        dist = msg.distance # distance of detected object (VERY NOISY)
        if dist > 0 and dist < self.vendor_min_dist and self.at_goal() and not self.dog_seen:
            rospy.loginfo("WooofWoof! A dog!")
            dist = dist*0.7

            theta_dog = (wrapToPi(msg.thetaleft) + wrapToPi(msg.thetaright))/2.
            theta_dog = theta_dog/2.
            # use occupancy grid to make sure location is inside boundary???
            location = Pose2D()
            location.x = self.x + dist*np.cos(self.theta + theta_dog)
            location.y = self.y + dist*np.sin(self.theta + theta_dog)
            #location.theta = self.theta + theta_vendor
            location.theta = 0.  # its a sphere, why complicate things 
            self.dog_location = location
            # add marker to rviz by publishing marker
            self.publish_dog_marker(msg)
            self.dog_seen = True


        
    def vendor_detected_callback(self,msg):
        dist = msg.distance # distance of detected object (VERY NOISY)
        vendor = msg.name # name of vendor detected

        if not self.vendor_locations.has_key(vendor):
            pass
	# elif self.mode == VendorMode.EXPLORE:
        elif self.mode == VendorMode.EXPLORE and self.at_goal():
            if dist > 0 and dist < self.vendor_min_dist and self.vendor_locations[vendor] is None:
                # Hacky fix to reduce distance
                dist = dist*0.7
                # save vendor locations
                theta_vendor = (wrapToPi(msg.thetaleft) + wrapToPi(msg.thetaright))/2.
                theta_vendor = theta_vendor/2.
                # use occupancy grid to make sure location is inside boundary???
                location = Pose2D()
                location.x = self.x + dist*np.cos(self.theta + theta_vendor)
                location.y = self.y + dist*np.sin(self.theta + theta_vendor)
                #location.theta = self.theta + theta_vendor
                location.theta = 0.  # its a sphere, why complicate things 
                self.vendor_locations[vendor] = location

                # update vendor nav locations
                nav_location = Pose2D()
                # nav_location.x = self.x + 0.5*dist*np.cos(self.theta + theta_vendor)
                nav_location.x = self.x
                # nav_location.y = self.y + 0.5*dist*np.cos(self.theta + theta_vendor)
                nav_location.y = self.y
                nav_location.theta = self.theta
                self.vendor_nav_locations[vendor] = nav_location

                # print statement
                rospy.loginfo("We saw vendor " + vendor + " and stored location")
                rospy.loginfo("Robot location: (" + str(self.x) + ", " + str(self.y) + ")")
                rospy.loginfo("Vendor location: (" + str(location.x) + ", " + str(location.y) + ")")
                rospy.loginfo("Theta robot: " + str(self.theta))
                rospy.loginfo("theta left: " + str(msg.thetaleft))
                rospy.loginfo("theta right: " + str(msg.thetaright))
                rospy.loginfo("theta_vendor: " + str(theta_vendor))
                rospy.loginfo("distance = " + str(dist))
                # add marker to rviz by publishing marker
                self.publish_marker(vendor,msg)


    # add delivery requests to list if not already there
    def delivery_request_callback(self,msg):
        message = msg.data
        if self.mode == VendorMode.HOME:
            self.switch_mode(VendorMode.DELIVER)
        # msg format "apple, hot_dog, banana"
        if len(self.delivery_requests) == 0 and self.mode == VendorMode.DELIVER:
            self.delivery_requests = message.split(',')
            # might want to add smarter reordering
            #self.delivery_requests.append(message)

    def goal_callback(self,data):
        # update goal
        if data.x != self.goal.x or data.y != self.goal.y or data.theta != self.goal.theta:
            self.goal = data # Pose2D

    def shutdown_callback(self):
        self.goal_pub.publish(self.home)

    ##### HELPER FUNCTIONS #####
    def publish_dog_marker(self, msg):
        # creates and publishes a marker for the dog
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time()
        marker.id = msg.id
        marker.type = 2 # sphere
        
        marker.pose.position.x = self.dog_location.x
        marker.pose.position.y = self.dog_location.y
        marker.pose.position.z = 0

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

	# sphere size
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2

        marker.color.a = 1.0 # Don't forget to set the alpha!
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        self.dog_marker_pub.publish(marker)

    def publish_marker(self, vendor,msg):
        # creates and publishes a marker for the vendor
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time()
        marker.id = msg.id
        marker.type = 2 # sphere
        
        marker.pose.position.x = self.vendor_locations[vendor].x
        marker.pose.position.y = self.vendor_locations[vendor].y
        marker.pose.position.z = 0

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

	# sphere size
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2

        marker.color.a = 1.0 # Don't forget to set the alpha!
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0

        self.vendor_marker_pub.publish(marker)

    def switch_mode(self, new_mode):
        # switch mode with print statement
        rospy.loginfo("EXPLIVER: switching from %s -> %s", self.mode, new_mode)
        self.mode = new_mode

    def at_goal(self):
        # returns True if we are at the goal position
        goal = self.goal
        return (linalg.norm(np.array([self.x-goal.x, self.y-goal.y])) < self.at_thresh and abs(wrapToPi(self.theta - goal.theta)) < self.at_thresh_theta)

    def init_vendor_stop(self):
        vendor = self.delivery_requests[0]
        # print that we're at the vendor
        rospy.loginfo("Reached vendor " + vendor + ", initializing vendor stop")
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
                self.goal = self.home
                self.goal_pub.publish(self.goal)
                if self.at_goal():
                    self.switch_mode(VendorMode.HOME)

            elif self.mode == VendorMode.DELIVER:
                # if there are no more vendors, go home
                if len(self.delivery_requests) == 0:
                    self.switch_mode(VendorMode.HOMING)
                else:
                    # set first entry in delivery_requests as the goal location
                    # rospy.loginfo("Setting new vendor goal " + self.delivery_requests[0])
                    self.goal = self.vendor_nav_locations[self.delivery_requests[0]]
                    # we should stop publishing goal when we're close
                    if not self.at_goal() and not self.vendor_stop:
                        self.goal_pub.publish(self.goal)
                    # check if we are at the vendor and haven't initialized stop
                    if self.at_goal() and not self.vendor_stop:
                        # start the vendor stop
                        self.init_vendor_stop()
                        self.vendor_stop = True
                    # if we've started the stop
                    if self.vendor_stop:
                        # if we've finished the stop (5 seconds)
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
                







