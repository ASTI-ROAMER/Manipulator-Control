#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64
import time
import numpy as np

'''
cd /home/albert/Desktop/trialSLAM/slam_test/src/roamer_husky/roamer_husky_description/scripts/
python3 mrmScript.py
roslaunch roamer_gazebo_husky husky_bp5_with_ekf.launch
'''


def talker():
    #/joint2_position_controller/command std_msgs/Float64 "data: 0.0"
    pub1 = rospy.Publisher('/joint1_position_controller/command', Float64, queue_size=10)
    pub2 = rospy.Publisher('/joint2_position_controller/command', Float64, queue_size=10)
    pub3 = rospy.Publisher('/joint3_position_controller/command', Float64, queue_size=10)
    pub4 = rospy.Publisher('/joint4_position_controller/command', Float64, queue_size=10)

    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        position1 = 0.0
        position2 = 0.0
        position3 = 0.0
        position4 = 0.0
        rospy.loginfo(position1)
        pub1.publish(position1)
        rospy.loginfo(position2)
        pub2.publish(position2)
        rospy.loginfo(position3)
        pub3.publish(position3)
        rospy.loginfo(position4)
        pub4.publish(position4)
        x_check = np.cos(position1) * (0.200*np.cos(position2) + 0.150*np.cos(position2+position3) + 0.072*np.cos(position2+position3+position4))
        y_check = np.sin(position1) * (0.200*np.cos(position2) + 0.150*np.cos(position2+position3) + 0.072*np.cos(position2+position3+position4))
        z_check = 0.150 + 0.200*np.sin(position2) + 0.150*np.sin(position2+position3) + 0.072*np.sin(position2+position3+position4)

        print('Result X: ' + str(x_check))
        print('Result Y: ' + str(y_check))
        print('Result Z: ' + str(z_check))

        time.sleep(2)

        position1 = 0.0
        position2 = 0.7854
        position3 = -1.57
        position4 = -0.7854
        rospy.loginfo(position1)
        pub1.publish(position1)
        rospy.loginfo(position2)
        pub2.publish(position2)
        rospy.loginfo(position3)
        pub3.publish(position3)
        rospy.loginfo(position4)
        pub4.publish(position4)
        x_check = np.cos(position1) * (0.200*np.cos(position2) + 0.150*np.cos(position2+position3) + 0.072*np.cos(position2+position3+position4))
        y_check = np.sin(position1) * (0.200*np.cos(position2) + 0.150*np.cos(position2+position3) + 0.072*np.cos(position2+position3+position4))
        z_check = 0.150 + 0.200*np.sin(position2) + 0.150*np.sin(position2+position3) + 0.072*np.sin(position2+position3+position4)

        print('Result X: ' + str(x_check))
        print('Result Y: ' + str(y_check))
        print('Result Z: ' + str(z_check))


        time.sleep(2)


        position1 = .0
        position2 = .0
        position3 = .0
        position4 = .0
        rospy.loginfo(position1)
        pub1.publish(position1)
        rospy.loginfo(position2)
        pub2.publish(position2)
        rospy.loginfo(position3)
        pub3.publish(position3)
        rospy.loginfo(position4)
        pub4.publish(position4)

        time.sleep(2)


        position1 = -1.57
        position2 = -0.7854
        position3 = 0.7854
        position4 = -0.7854
        rospy.loginfo(position1)
        pub1.publish(position1)
        rospy.loginfo(position2)
        pub2.publish(position2)
        rospy.loginfo(position3)
        pub3.publish(position3)
        rospy.loginfo(position4)
        pub4.publish(position4)
        x_check = np.cos(position1) * (0.200*np.cos(position2) + 0.150*np.cos(position2+position3) + 0.072*np.cos(position2+position3+position4))
        y_check = np.sin(position1) * (0.200*np.cos(position2) + 0.150*np.cos(position2+position3) + 0.072*np.cos(position2+position3+position4))
        z_check = 0.150 + 0.200*np.sin(position2) + 0.150*np.sin(position2+position3) + 0.072*np.sin(position2+position3+position4)

        print('Result X: ' + str(x_check))
        print('Result Y: ' + str(y_check))
        print('Result Z: ' + str(z_check))

        time.sleep(2)


        position1 = 1.57
        position2 = -2.3561
        position3 = 2.3561
        position4 = -0.7854        
        rospy.loginfo(position1)
        pub1.publish(position1)
        rospy.loginfo(position2)
        pub2.publish(position2)
        rospy.loginfo(position3)
        pub3.publish(position3)
        rospy.loginfo(position4)
        pub4.publish(position4)
        x_check = np.cos(position1) * (0.200*np.cos(position2) + 0.150*np.cos(position2+position3) + 0.072*np.cos(position2+position3+position4))
        y_check = np.sin(position1) * (0.200*np.cos(position2) + 0.150*np.cos(position2+position3) + 0.072*np.cos(position2+position3+position4))
        z_check = 0.150 + 0.200*np.sin(position2) + 0.150*np.sin(position2+position3) + 0.072*np.sin(position2+position3+position4)

        print('Result X: ' + str(x_check))
        print('Result Y: ' + str(y_check))
        print('Result Z: ' + str(z_check))

        time.sleep(2)

        position1 = 0.0
        position2 = 0.0
        position3 = 0.0
        position4 = 0.0
        rospy.loginfo(position1)
        pub1.publish(position1)
        rospy.loginfo(position2)
        pub2.publish(position2)
        rospy.loginfo(position3)
        pub3.publish(position3)
        rospy.loginfo(position4)
        pub4.publish(position4)
        x_check = np.cos(position1) * (0.200*np.cos(position2) + 0.150*np.cos(position2+position3) + 0.072*np.cos(position2+position3+position4))
        y_check = np.sin(position1) * (0.200*np.cos(position2) + 0.150*np.cos(position2+position3) + 0.072*np.cos(position2+position3+position4))
        z_check = 0.150 + 0.200*np.sin(position2) + 0.150*np.sin(position2+position3) + 0.072*np.sin(position2+position3+position4)

        print('Result X: ' + str(x_check))
        print('Result Y: ' + str(y_check))
        print('Result Z: ' + str(z_check))



        '''

        rostopic pub /joint2_position_controller/command std_msgs/Float64 "data: 0.7854"

        rostopic pub /joint3_position_controller/command std_msgs/Float64 "data: -1.57"

        rostopic pub /joint4_position_controller/command std_msgs/Float64 "data: -0.7854"


        rostopic pub /joint2_position_controller/command std_msgs/Float64 "data: 0.0"

        rostopic pub /joint3_position_controller/command std_msgs/Float64 "data: 0.0"

        rostopic pub /joint4_position_controller/command std_msgs/Float64 "data: 0.0"


        rostopic pub /joint1_position_controller/command std_msgs/Float64 "data: -1.57"

        rostopic pub /joint2_position_controller/command std_msgs/Float64 "data: -0.7854"

        rostopic pub /joint3_position_controller/command std_msgs/Float64 "data: 0.7854"

        rostopic pub /joint4_position_controller/command std_msgs/Float64 "data: -0.7854"




        rostopic pub /joint2_position_controller/command std_msgs/Float64 "data: -2.3561"

        rostopic pub /joint3_position_controller/command std_msgs/Float64 "data: 2.3561"

        rostopic pub /joint4_position_controller/command std_msgs/Float64 "data: -0.7854"

        rostopic pub /joint1_position_controller/command std_msgs/Float64 "data: 1.57"

        '''



        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass