#!/usr/bin/env python

import numpy as np
from utils import wrapToPi

# Import message definition
import rospy 
from std_msgs.msg import Float32

# command zero velocities once we are this close to the goal
RHO_THRES = 0.05
ALPHA_THRES = 0.1
DELTA_THRES = 0.1

class PoseController:
    """ Pose stabilization controller """
    def __init__(self, k1, k2, k3, V_max=0.5, om_max=1):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

        self.V_max = V_max
        self.om_max = om_max


	# Define Publishers 
	self.alpha_pub = rospy.Publisher('/controller/alpha', Float32, queue_size=10)
	self.rho_pub = rospy.Publisher('/controller/rho', Float32, queue_size=10)
	self.delta_pub = rospy.Publisher('/controller/delta', Float32, queue_size=10)
	
	# Init the node 
	#rospy.init_node('PoseControlNode', anonymous=True)




    def load_goal(self, x_g, y_g, th_g):
        """ Loads in a new goal position """
        self.x_g = x_g
        self.y_g = y_g
        self.th_g = th_g

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            x,y,th: Current state
            t: Current time (you shouldn't need to use this)
        Outputs: 
            V, om: Control actions

        Hints: You'll need to use the wrapToPi function. The np.sinc function
        may also be useful, look up its documentation
        """
        ########## Code starts here ##########
        rho = np.sqrt((x - self.x_g)**2 + (y - self.y_g)**2)
        goal_heading = np.arctan2(self.y_g - y, self.x_g - x)
        alpha = wrapToPi(goal_heading - th)
        delta = wrapToPi(goal_heading - self.th_g) 

	# Publish rho, delta, alpha 
	self.alpha_pub.publish(alpha)
	self.rho_pub.publish(rho)
	self.delta_pub.publish(delta)

        
        if ((np.abs(alpha) < ALPHA_THRES) and (np.abs(delta) < DELTA_THRES))\
                            and (rho < RHO_THRES):
            om = 0  # stay put 
            V = 0   
        else: 
            om = self.k2*alpha + self.k1*(np.cos(alpha)*np.sinc(alpha/np.pi))*(alpha+self.k3*delta)
            V = self.k1*rho*np.cos(alpha)
        



        ########## Code ends here ##########

        # apply control limits
        V = np.clip(V, -self.V_max, self.V_max)
        om = np.clip(om, -self.om_max, self.om_max)

        return V, om
