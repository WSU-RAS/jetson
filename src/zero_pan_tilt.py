#!/usr/bin/env python2
import rospy
from std_msgs.msg import Float64

def panTilt():
    pan = rospy.Publisher('/head_pan_joint/command', Float64, queue_size=2)
    tilt = rospy.Publisher('/head_tilt_joint/command', Float64, queue_size=2)
    rospy.init_node('pan_tilt', anonymous=True)
    hz = 10
    seconds = 10 # exit after this many seconds (takes some time for Arbotix to launch)
    rate = rospy.Rate(hz) # Hz

    i = 0
    while not rospy.is_shutdown():
        pan.publish(0)
        #tilt.publish(0.25)
        tilt.publish(0.7)
        rate.sleep()

        # Run only briefly
        i += 1
        if i >hz*seconds: 
            break

if __name__ == '__main__':
    try:
        panTilt()
    except rospy.ROSInterruptException:
        pass
