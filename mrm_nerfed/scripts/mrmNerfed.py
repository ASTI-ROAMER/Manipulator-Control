#!/usr/bin/env python
import time
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
'''
cd /home/albert/Desktop/trialSLAM/slam_test/src/roamer_husky/roamer_husky_description/scripts/
python3 mrmScript.py
roslaunch roamer_gazebo_husky husky_bp5_with_ekf.launch
'''


def mover():
    pub = rospy.Publisher('joint_states', JointState, queue_size=10)
    rospy.init_node('joint_state_publisher')
    rate = rospy.Rate(10) # 10hz
    command_position = JointState()
    command_position.header = Header()
    command_position.header.stamp = rospy.Time.now()
    command_position.name = ['link_1_joint', 'link_2_joint', 'link_3_joint', 'link_4_joint']
    command_position.position = [0.0, 0.0, 0.0, 0.0]
    command_position.velocity = []
    command_position.effort = []


    while not rospy.is_shutdown():

        #Nest ----------------------------------------------------------------------------------------------------
        pos1 = 0.0
        pos2 = 0.0
        pos3 = 0.0
        pos4 = 0.0
        command_position.header.stamp = rospy.Time.now()
        command_position.position = [pos1, pos2, pos3, pos4]
        pub.publish(command_position)

        rospy.loginfo(pos1)
        rospy.loginfo(pos2)
        rospy.loginfo(pos3)
        rospy.loginfo(pos4)
        x_check = np.cos(pos1) * (0.300*np.cos(pos2) + 0.305*np.cos(pos2+pos3) + 0.071*np.cos(pos2+pos3+pos4))
        y_check = np.sin(pos1) * (0.300*np.cos(pos2) + 0.305*np.cos(pos2+pos3) + 0.071*np.cos(pos2+pos3+pos4))
        z_check = 0.250 + 0.300*np.sin(pos2) + 0.305*np.sin(pos2+pos3) + 0.071*np.sin(pos2+pos3+pos4)

        print('Nest')
        print('Result X: ' + str(x_check))
        print('Result Y: ' + str(y_check))
        print('Result Z: ' + str(z_check))
        
        '''
        Result X: 0.6759999999999999
        Result Y: 0.0
        Result Z: 0.25
        '''

        time.sleep(2)

        #Stand ----------------------------------------------------------------------------------------------------
        pos1 = 1.01
        pos2 = 1.78
        pos3 = 0.0
        pos4 = -1.57
        command_position.header.stamp = rospy.Time.now()
        command_position.position = [pos1, pos2, pos3, pos4]
        pub.publish(command_position)

        rospy.loginfo(pos1)
        rospy.loginfo(pos2)
        rospy.loginfo(pos3)
        rospy.loginfo(pos4)
        x_check = np.cos(pos1) * (0.300*np.cos(pos2) + 0.305*np.cos(pos2+pos3) + 0.071*np.cos(pos2+pos3+pos4))
        y_check = np.sin(pos1) * (0.300*np.cos(pos2) + 0.305*np.cos(pos2+pos3) + 0.071*np.cos(pos2+pos3+pos4))
        z_check = 0.250 + 0.300*np.sin(pos2) + 0.305*np.sin(pos2+pos3) + 0.071*np.sin(pos2+pos3+pos4)

        print('Stand')
        print('Result X: ' + str(x_check))
        print('Result Y: ' + str(y_check))
        print('Result Z: ' + str(z_check))

        '''
        Result X: -0.029894195059450807
        Result Y: -0.0475977174628525
        Result Z: 0.8566096000079401

        '''

        time.sleep(2)

        #Left Under ----------------------------------------------------------------------------------------------------
        pos1 = 1.01
        pos2 = 1.85
        pos3 = -2.35
        pos4 = 1.15
        command_position.header.stamp = rospy.Time.now()
        command_position.position = [pos1, pos2, pos3, pos4]
        pub.publish(command_position)

        rospy.loginfo(pos1)
        rospy.loginfo(pos2)
        rospy.loginfo(pos3)
        rospy.loginfo(pos4)
        x_check = np.cos(pos1) * (0.300*np.cos(pos2) + 0.305*np.cos(pos2+pos3) + 0.071*np.cos(pos2+pos3+pos4))
        y_check = np.sin(pos1) * (0.300*np.cos(pos2) + 0.305*np.cos(pos2+pos3) + 0.071*np.cos(pos2+pos3+pos4))
        z_check = 0.250 + 0.300*np.sin(pos2) + 0.305*np.sin(pos2+pos3) + 0.071*np.sin(pos2+pos3+pos4)
        x_check2 = np.cos(pos1) * (0.300*np.cos(pos2) + 0.305*np.cos(pos2+pos3))
        y_check2 = np.sin(pos1) * (0.300*np.cos(pos2) + 0.305*np.cos(pos2+pos3))
        z_check2 = 0.250 + 0.300*np.sin(pos2) + 0.305*np.sin(pos2+pos3)

        print('Left Under')
        print('Result X: ' + str(x_check))
        print('Result Y: ' + str(y_check))
        print('Result Z: ' + str(z_check))
        print('Result link3_X: ' + str(x_check2))
        print('Result link3_Y: ' + str(y_check2))
        print('Result link3_Z: ' + str(z_check2))


        '''
        Result X: 0.1284483834973092
        Result Y: 0.20451629000568133
        Result Z: 0.4351260064255668
        '''

        time.sleep(2)


        #Rest ----------------------------------------------------------------------------------------------------
        pos1 = 0.05
        pos2 = 3.14
        pos3 = -2.97
        pos4 = 1.57
        command_position.header.stamp = rospy.Time.now()
        command_position.position = [pos1, pos2, pos3, pos4]
        pub.publish(command_position)

        rospy.loginfo(pos1)
        rospy.loginfo(pos2)
        rospy.loginfo(pos3)
        rospy.loginfo(pos4)
        x_check = np.cos(pos1) * (0.300*np.cos(pos2) + 0.305*np.cos(pos2+pos3) + 0.071*np.cos(pos2+pos3+pos4))
        y_check = np.sin(pos1) * (0.300*np.cos(pos2) + 0.305*np.cos(pos2+pos3) + 0.071*np.cos(pos2+pos3+pos4))
        z_check = 0.250 + 0.300*np.sin(pos2) + 0.305*np.sin(pos2+pos3) + 0.071*np.sin(pos2+pos3+pos4)

        print('Rest')
        print('Result X: ' + str(x_check))
        print('Result Y: ' + str(y_check))
        print('Result Z: ' + str(z_check))

        '''
        Result X: -0.011338296765840994
        Result Y: -0.0005673877402315296
        Result Z: 0.3720644740377041
        '''

        time.sleep(2)


        #Left Front ----------------------------------------------------------------------------------------------------
        pos1 = 1.20
        pos2 = 2.12
        pos3 = -2.09
        pos4 = 0.14
        command_position.header.stamp = rospy.Time.now()
        command_position.position = [pos1, pos2, pos3, pos4]
        pub.publish(command_position)

        rospy.loginfo(pos1)
        rospy.loginfo(pos2)
        rospy.loginfo(pos3)
        rospy.loginfo(pos4)
        x_check = np.cos(pos1) * (0.300*np.cos(pos2) + 0.305*np.cos(pos2+pos3) + 0.071*np.cos(pos2+pos3+pos4))
        y_check = np.sin(pos1) * (0.300*np.cos(pos2) + 0.305*np.cos(pos2+pos3) + 0.071*np.cos(pos2+pos3+pos4))
        z_check = 0.250 + 0.300*np.sin(pos2) + 0.305*np.sin(pos2+pos3) + 0.071*np.sin(pos2+pos3+pos4)

        print('Left Front')
        print('Result X: ' + str(x_check))
        print('Result Y: ' + str(y_check))
        print('Result Z: ' + str(z_check))

        '''
        Result X: 0.07907980628508124
        Result Y: 0.20340525201360676
        Result Z: 0.5270427188113808

        '''

        time.sleep(2)


        #Right Back Under ----------------------------------------------------------------------------------------------------
        pos1 = -2.09
        pos2 = 0.71
        pos3 = -1.19
        pos4 = 0.81
        command_position.header.stamp = rospy.Time.now()
        command_position.position = [pos1, pos2, pos3, pos4]
        pub.publish(command_position)

        rospy.loginfo(pos1)
        rospy.loginfo(pos2)
        rospy.loginfo(pos3)
        rospy.loginfo(pos4)
        x_check = np.cos(pos1) * (0.300*np.cos(pos2) + 0.305*np.cos(pos2+pos3) + 0.071*np.cos(pos2+pos3+pos4))
        y_check = np.sin(pos1) * (0.300*np.cos(pos2) + 0.305*np.cos(pos2+pos3) + 0.071*np.cos(pos2+pos3+pos4))
        z_check = 0.250 + 0.300*np.sin(pos2) + 0.305*np.sin(pos2+pos3) + 0.071*np.sin(pos2+pos3+pos4)

        print('Right Back Under')
        print('Result X: ' + str(x_check))
        print('Result Y: ' + str(y_check))
        print('Result Z: ' + str(z_check))

        '''
        Result X: -0.28045144177850195
        Result Y: -0.4907244508397896
        Result Z: 0.3277145377823444

        '''

        time.sleep(2)


        rate.sleep() 


if __name__ == '__main__':
    try:
        mover()
    except rospy.ROSInterruptException:
        pass