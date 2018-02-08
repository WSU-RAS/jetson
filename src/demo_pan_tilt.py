#!/usr/bin/env python2
"""
Example code to publish directly to head pan/tilt joint states

This will make the camera pan/tilt perform some back and forth motion for
testing that the model correctly follows the real movement in rviz.
"""
import rospy
from std_msgs.msg import Float64

def panTilt():
    pan = rospy.Publisher('/head_pan_joint/command', Float64, queue_size=2)
    tilt = rospy.Publisher('/head_tilt_joint/command', Float64, queue_size=2)
    rospy.init_node('pan_tilt')
    rate = rospy.Rate(10) # 10hz

    while not rospy.is_shutdown():
        for i in range(100):
            pan.publish(1.0*(i-50)/100)
            tilt.publish(1.0*i/100/2)
            rate.sleep()

        for i in range(100):
            j = 100-i
            pan.publish(1.0*(j-50)/100)
            tilt.publish(1.0*j/100/2)
            rate.sleep()

if __name__ == '__main__':
    try:
        panTilt()
    except rospy.ROSInterruptException:
        pass
