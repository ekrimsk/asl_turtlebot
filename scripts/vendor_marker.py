#!/usr/bin/env python

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose2D


class MarkerPub:
    """
    This node handles point to point turtlebot motion, avoiding obstacles.
    It is the sole node that should publish to cmd_vel
    """
    def __init__(self):

        self.marker = Marker()

        self.vis_pub = rospy.Publisher('marker_topic', Marker, queue_size=10)
        rospy.init_node('marker_node', anonymous=True)
        rospy.Subscriber('/cmd_nav', Pose2D, self.cmd_nav_callback)

        self.rate = rospy.Rate(1)

    def cmd_nav_callback(self, data):
        """
        loads in goal if different from current goal, and replans
        """
        marker = self.marker
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time()

        # IMPORTANT: If you're creating multiple markers, 
        # each need to have a separate marker ID.
        marker.id = 0

        marker.type = 2 # sphere

        marker.pose.position.x = data.x
        marker.pose.position.y = data.y
        marker.pose.position.z = 0

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        marker.color.a = 1.0 # Don't forget to set the alpha!
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        self.marker = marker


    def marker_pub(self):
        self.vis_pub.publish(self.marker)
        # print('Published marker!')
        self.rate.sleep()
            

if __name__ == '__main__':
    try:
        mrk = MarkerPub()
    	while not rospy.is_shutdown():
            mrk.marker_pub()
    except rospy.ROSInterruptException:
        pass


