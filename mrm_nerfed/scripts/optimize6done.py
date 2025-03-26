import numpy as np
import math
import time
import cv2
import rospy
import matplotlib.pyplot as plt
from scipy.integrate import odeint #ode64 matlab
from scipy.optimize import minimize #fmincon matlab

from std_msgs.msg import Float64
from sensor_msgs.msg import JointState

import time


import faulthandler
faulthandler.enable()

'''
cd /home/albert/Desktop/trialSLAM/slam_test/src/roamer_husky/roamer_description_husky/scripts/
python3 mrmScript.py
roslaunch roamer_gazebo_husky husky_bp5_with_ekf.launch

cd /home/albert/Desktop/trialSLAM/slam_test/src/roamer_husky/roamer_husky_description/scripts/
python3 optimize6.py
roslaunch roamer_husky_mbf_nav bpsim2_rtab_ekf_odom_arm.launch
rosrun teleop_twist_keyboard teleop_twist_keyboard.py

rosparam set use_sim_time true
'''


# End Effector coordinates and joint angles
x = 0.0
y = 0.0
z = 0.0
q_Current = [0.0, 0.0, 0.0, 0.0]


def cost(q): # Cost function for joint angles
	#print("costFunction")

	global x_link3
	global y_link3
	global z_link3

	global x_endEffector
	global y_endEffector
	global z_endEffector

	q1 = q[0]
	q2 = q[1]
	q3 = q[2]
	q4 = q[3]

	#xd = np.cos(q1)*(0.085*np.sin(q2+q3+q4) + 0.150*np.cos(q2+q3) +0.200*np.cos(q2))
	#yd = np.sin(q1)*(0.085*np.sin(q2+q3+q4) + 0.150*np.cos(q2+q3) +0.200*np.cos(q2))
	#zd = 0.150-0.085*np.cos(q2+q3+q4) + 0.150*np.sin(q2+q3) +0.200*np.sin(q2)

	#l1 = np.cos(q1) or np.sin(q1)
	#l2 = 0.200*np.cos(q2)
	#l3 = 0.150*np.cos(q2+q3)
	#l4 = 0.085*np.cos(q2+q3+q4)

	xd = np.cos(q1) * (0.200*np.cos(q2) + 0.150*np.cos(q2+q3) + 0.085*np.cos(q2+q3+q4))
	yd = np.sin(q1) * (0.200*np.cos(q2) + 0.150*np.cos(q2+q3) + 0.085*np.cos(q2+q3+q4))
	zd = 0.150 + 0.200*np.sin(q2) + 0.150*np.sin(q2+q3) + 0.085*np.sin(q2+q3+q4)

	xa = np.cos(q1) * (0.200*np.cos(q2) + 0.150*np.cos(q2+q3))
	ya = np.sin(q1) * (0.200*np.cos(q2) + 0.150*np.cos(q2+q3))
	za = 0.150 + 0.200*np.sin(q2) + 0.150*np.sin(q2+q3)

	c = (xd-x_endEffector)**2 + (yd-y_endEffector)**2 + (zd-z_endEffector)**2 #Minimize distance
	c = c + (xa-x_link3)**2 + (ya-y_link3)**2 + (za-z_link3)**2 #Minimize distance

	return c

def getJointStates(message): # https://answers.ros.org/question/260430/how-can-i-write-a-node-to-subscribe-to-the-joint-0/
	print("getJointStates")

	'''
	rostopic echo joint_states

	header: 
	  seq: 270016
	  stamp: 
		secs: 2980
		nsecs: 554000000
	  frame_id: ''
	name: 
	  - front_left_wheel
	  - front_right_wheel
	  - link_1_joint
	  - link_2_joint
	  - link_3_joint
	  - link_4_joint
	  - rear_left_wheel
	  - rear_right_wheel
	'''

	global q_Current
	global x_Current
	global y_Current
	global z_Current

	global x_CurrentLink3
	global y_CurrentLink3
	global z_CurrentLink3

	for i, name in enumerate(message.name):
		if name == "link_1_joint": # <!-- for Base to Link 1 --> mrm.urdf.xacro
			q_Current[0] = message.position[i]
		elif name == "link_2_joint": # <!-- for Link 1 to Link 2 --> 
			q_Current[1] = message.position[i]
		elif name == "link_3_joint": # <!-- for Link 2 to Link 3 --> 
			q_Current[2] = message.position[i]
		elif name == "link_4_joint": # <!-- for Link 3 to Link 4 --> 
			q_Current[3] = message.position[i]
		else:
			#print("passed ")
			#print(name)
			pass

	x_Current = np.cos(q_Current[0]) * (0.200*np.cos(q_Current[1]) + 0.150*np.cos(q_Current[1]+q_Current[2]) + 0.085*np.cos(q_Current[1]+q_Current[2]+q_Current[3]))
	y_Current = np.sin(q_Current[0]) * (0.200*np.cos(q_Current[1]) + 0.150*np.cos(q_Current[1]+q_Current[2]) + 0.085*np.cos(q_Current[1]+q_Current[2]+q_Current[3]))
	z_Current = 0.150 + 0.200*np.sin(q_Current[1]) + 0.150*np.sin(q_Current[1]+q_Current[2]) + 0.085*np.sin(q_Current[1]+q_Current[2]+q_Current[3])


	x_CurrentLink3 = np.cos(q_Current[0]) * (0.200*np.cos(q_Current[1]) + 0.150*np.cos(q_Current[1]+q_Current[2]) )
	y_CurrentLink3 = np.sin(q_Current[0]) * (0.200*np.cos(q_Current[1]) + 0.150*np.cos(q_Current[1]+q_Current[2]) )
	z_CurrentLink3 = 0.150 + 0.200*np.sin(q_Current[1]) + 0.150*np.sin(q_Current[1]+q_Current[2]) 


	'''
	print('State Angle 1: ' + str(q_Current[0]))
	print('State Angle 2: ' + str(q_Current[1]))
	print('State Angle 3: ' + str(q_Current[2]))
	print('State Angle 4: ' + str(q_Current[3]))
	'''

	return


def solveAnglesfromXYZ():
	print("solveAnglesfromXYZ")

	global x_test
	global y_test
	global z_test

	global x_link3
	global y_link3
	global z_link3

	global x_endEffector
	global y_endEffector
	global z_endEffector


	# Set sphere radius
	sphereRadius = 0.435 # meters l1 = 0.150 (not included), l2 = 0.200 + l3 = 150 + l4 = 0.085 = 0.435m
	sphereCenter = np.array([0, 0, 0.150])

	# Check if within sphere (max radius of working space)
	checkArmInSphere = (x_test - sphereCenter[0])**2 + (y_test - sphereCenter[1])**2 + (z_test - sphereCenter[2])**2

	if checkArmInSphere <= sphereRadius**2: # Setpoint is within sphere
		# Given xyx_test setpoint for the end effector, get the setpoint for link 3 along the line between the sphere center and the setpoint.
		#	1. Subtract point from sphere center to get vector from the center sphere to the point
		goalPoint_inSphere = [x_test, y_test, z_test] #******************
		vectorGoal = np.array(goalPoint_inSphere)
		vectorDiff = np.subtract(vectorGoal, sphereCenter) #per element subtraction

		# 	2. Normalize vector to get direction (unit vector) 
		vectorNorm = vectorDiff/np.linalg.norm(vectorDiff)

		#	3. The point along the vector at a distance d (link 3's length) from the setpoint is #https://math.stackexchange.com/a/175906 https://stackoverflow.com/a/11775520
		lengthLink3 = 0.150
		d_prime = np.sqrt(checkArmInSphere)-lengthLink3 #distance from startpoint to link3 endpoint
		goalPoint_link3 = np.add(vectorNorm*(d_prime), sphereCenter) #******************

		x_link3 = goalPoint_link3[0]
		y_link3 = goalPoint_link3[1]
		z_link3 = goalPoint_link3[2]

		x_endEffector = goalPoint_inSphere[0]
		y_endEffector = goalPoint_inSphere[1]
		z_endEffector = goalPoint_inSphere[2]


	else: #Not within sphere
		# Get closest point from setpoint (xyz_test) on tree to the point on the sphere's surface
		#	1. Subtract point from sphere center to get vector from the center sphere to the point
		list_test = [x_test, y_test, z_test]
		vectorGoal = np.array(list_test)
		vectorDiff = np.subtract(vectorGoal, sphereCenter) #per element subtraction

		# 	2. Normalize vector to get direction (unit vector) then multiply it to the sphere radius -> to get vector from sphere center to the edge of the sphere towards the point
		vectorNorm = vectorDiff/np.linalg.norm(vectorDiff)
		
		#	3. Add the position of the sphere as a vector to get the vector that points from the origin of the arm_base_link ("world") to the edge of the sphere (ie: add l1 = 0.150m at the z-axis)
		goalPoint_onSphere = np.add(vectorNorm*sphereRadius, sphereCenter) #******************

		#	4. Subtract point from sphere center to get vector from the center sphere to the point on edge of sphere
		vectorGoal = np.array(goalPoint_onSphere)
		vectorDiff = np.subtract(vectorGoal, sphereCenter) #per element subtraction

		# 	5. Normalize vector to get direction (unit vector) 
		vectorNorm = vectorDiff/np.linalg.norm(vectorDiff)

		#	6. The point along the vector at a distance d (link 3's length) from the setpoint is #https://math.stackexchange.com/a/175906 https://stackoverflow.com/a/11775520
		lengthLink3 = 0.150
		d_prime = np.sqrt(checkArmInSphere)-lengthLink3 #distance from startpoint to link3 endpoint
		goalPoint_link3 = np.add(vectorNorm*(d_prime), sphereCenter) #******************

		x_link3 = goalPoint_link3[0]
		y_link3 = goalPoint_link3[1]
		z_link3 = goalPoint_link3[2]

		x_endEffector = goalPoint_onSphere[0]
		y_endEffector = goalPoint_onSphere[1]
		z_endEffector = goalPoint_onSphere[2]


		'''

		# Get the x-axis normal unit vector on the point on the sphere (unit vector at <x_test, y_test, z_test> on x^2 + y^2 + (z-0.150)^2 - (0.435)^2 = 0 or F(x,y,z) = 0 )
		#	1. Get gradient
		vectorGrad = np.array([2*x_sphere, 2*y_sphere, 2*(z_sphere-0.150)])
		
		#	2. Normalize
		vectorNorm_X = vectorGrad/np.linalg.norm(vectorGrad)

		# Get the y-axis unit vector (y-axis is always parallel to the xy plane)
		#1. Consider this, x-axis unit vector turns into x_ihat + y_jhat + z_khat and othrogonal vector turns to a_ihat + b_jhat + 0_khat. If x-axis normal vector is parallel to xy-plane, x-axis unit vector * y-axis unit vector becomes:
		#	x*a_ihat + y*b_jhat + z*0_khat = 0 - their dot product must be zero if perpendicular
		#	x*a = -y*b
		#	a = -y*b/x & b = -x*a/y
		# 	a:b = x:-y

		#2. Let a = x*n, b = -y*n so u_y = x*n_ihat - y*n_jhat, therefore the magnitude of u_y:
		#	|u_y| = sqrt((x*n)^2+(-y_n)^2)
		#		  = n*sqrt(x^2+y^2)
		#	and,
		#	norm(u_y) = u_y/|u_y| = x/sqrt(x^2+y^2)_ihat - y/sqrt(x^2+y^2)_jhat

		vectorMag_Y = np.sqrt(vectorNorm_X[0]**2 + vectorNorm_X[1]**2)
		vectorNorm_Y = np.array([vectorNorm_X[0]/vectorMag_Y, vectorNorm_X[1]/vectorMag_Y, 0])

		# Get the z_axis unit vector by getting the cross product of the x and y unit vectors
		vectorNorm_Z = np.cross(vectorNorm_X, vectorNorm_Y)

		'''

	# Upper bound and lower bound limits of each joint angle in degrees based from RV-M1: https://www.researchgate.net/figure/Mitsubishis-Movemaster-RV-M2-robot-configuration-and-dimensions_fig2_326399731
	q_upperBound = [180, 110, 120, 120] #[180, 110, 120, 120] 
	q_lowerBound = [-180, -20, -120, -120] #[-180, -20, -120, -120]
	q_allBounds = ((np.deg2rad(-180),np.deg2rad(180)),(np.deg2rad(-20), np.deg2rad(110)),(np.deg2rad(-120),np.deg2rad(120)),(np.deg2rad(-120),np.deg2rad(120)))

	# fmincon(fun,x0,A,b,Aeq,beq,lb,ub) #Matlab to Scipy
	# fun - function to minimize
	# x0 - initial point
	# A - linear inequality constraints M-by-N matrix, M - number of inequalities, N - number of elements in x0
	# b - another linear inequality constrain Ax <= b
	# Aeq - linear equality constraints
	# beq
	# lb - lower bound
	# ub - upper bound
	
	# Input initial state and constraints 
	global q_Current
	global x_Current
	global y_Current
	global z_Current

	global x_CurrentLink3
	global y_CurrentLink3
	global z_CurrentLink3

	q_initialAngles = [q_Current[0], q_Current[1], q_Current[2], q_Current[3]]
	#A = #[0,1,1,0; 0 0 1 1] #make sure not to hit goal post/lidar
	#b = #[180;180]
	#Aeq = []
	#beq = []
	print('Current Angle 1: ' + str(q_Current[0]))
	print('Current Angle 2: ' + str(q_Current[1]))
	print('Current Angle 3: ' + str(q_Current[2]))
	print('Current Angle 4: ' + str(q_Current[3]))

	print('Current X: ' + str(x_Current))
	print('Current Y: ' + str(y_Current))
	print('Current Z: ' + str(z_Current))

	print('Current Link3 X: ' + str(x_CurrentLink3))
	print('Current Link3 Y: ' + str(y_CurrentLink3))
	print('Current Link3 Z: ' + str(z_CurrentLink3))

	# Minimize function using objective equation c from cost.py
	print("costFunction")
	result = minimize(cost, q_initialAngles, method='SLSQP', bounds = q_allBounds) #scipy.optimize.minimize
	print('SSE Objective: ' + str(cost(result.x))) #Prints optimization result

	q_Results = result.x #-> solution is stored in .x object 
	print('Result Angle 1: ' + str(q_Results[0]))
	print('Result Angle 2: ' + str(q_Results[1]))
	print('Result Angle 3: ' + str(q_Results[2]))
	print('Result Angle 4: ' + str(q_Results[3]))

	x_check = np.cos(q_Results[0]) * (0.200*np.cos(q_Results[1]) + 0.150*np.cos(q_Results[1]+q_Results[2]) + 0.085*np.cos(q_Results[1]+q_Results[2]+q_Results[3]))
	y_check = np.sin(q_Results[0]) * (0.200*np.cos(q_Results[1]) + 0.150*np.cos(q_Results[1]+q_Results[2]) + 0.085*np.cos(q_Results[1]+q_Results[2]+q_Results[3]))
	z_check = 0.150 + 0.200*np.sin(q_Results[1]) + 0.150*np.sin(q_Results[1]+q_Results[2]) + 0.085*np.sin(q_Results[1]+q_Results[2]+q_Results[3])

	print('Result X: ' + str(x_check))
	print('Result Y: ' + str(y_check))
	print('Result Z: ' + str(z_check))

	# Publish joint angles/commands
	rospy.loginfo(q_Results[0])
	pub1.publish(q_Results[0])
	rospy.loginfo(q_Results[1])
	pub2.publish(q_Results[1])
	rospy.loginfo(q_Results[2])
	pub3.publish(q_Results[2])
	rospy.loginfo(q_Results[3])
	pub4.publish(q_Results[3])

	time.sleep(10)


	return


if __name__ == '__main__':

	# Input end effector destination (Variable) from trajectory (convert first from camera frame to base link)
	'''
	x_test = -0.16 #from Screenshot from 2023-05-04 15-41-33.png (base link)
	y_test = 0.391
	z_test = 0.603
	q1_key = -1.57
	q2_key = -0.78754
	q3_key = 0.78754
	q4_key = -0.78754
	'''

	#BaseLink
	'''
	x_test = -0.427
	y_test = -0.00413
	z_test = 0.451
	q1_key = 0.0 #5.079737083413249e-06
	q2_key = 0.7854#0.7958387377357639
	q3_key = -1.57#-1.5542413486650397
	q4_key = -0.7854#-0.7854000000000187
	'''

	#ArmBaseLink
	
	x_test = 4.7384307709970994e-05#-0.000171204362945983
	y_test = 0.05950358319926715#0.05934423389513097
	z_test = -0.04234650028699127#-0.04220394513709968
	q1_key = 1.57#1.5736812555696265
	q2_key = -2.3561#-0.1602308306025748
	q3_key = 2.3561#-1.9118479063968554
	q4_key = -0.7854#-0.658759567133252
	

	try:
		rospy.init_node("robotArm")
		
		pub1 = rospy.Publisher('/joint1_position_controller/command', Float64, queue_size=10)
		pub2 = rospy.Publisher('/joint2_position_controller/command', Float64, queue_size=10)
		pub3 = rospy.Publisher('/joint3_position_controller/command', Float64, queue_size=10)
		pub4 = rospy.Publisher('/joint4_position_controller/command', Float64, queue_size=10)
		
		sub1 = rospy.Subscriber("joint_states", JointState, getJointStates, queue_size=10)
		

		rate = rospy.Rate(float(rospy.get_param('~rate', 30.0)))
		
		while not rospy.is_shutdown(): #https://answers.ros.org/question/310291/while-loop-only-executes-subscriber-callback/
			#	Nest 
			print("Nest")
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

			time.sleep(10)

			print("Nest")
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

			time.sleep(10)

			'''
			print("Test")
			position1 = 0.0
			position2 = 0.7854
			position3 = -1.57
			position4 = -0.7854
			x_key = np.cos(q1_key) * (0.200*np.cos(q2_key) + 0.150*np.cos(q2_key+q3_key) + 0.085*np.cos(q2_key+q3_key+q4_key))
			y_key = np.sin(q1_key) * (0.200*np.cos(q2_key) + 0.150*np.cos(q2_key+q3_key) + 0.085*np.cos(q2_key+q3_key+q4_key))
			z_key = 0.150 + 0.200*np.sin(q2_key) + 0.150*np.sin(q2_key+q3_key) + 0.085*np.sin(q2_key+q3_key+q4_key)
			
			print('Key X: ' + str(x_key))
			print('Key Y: ' + str(y_key))
			print('Key Z: ' + str(z_key))

			#Test
			#Key X: 0.2476290734215763
			#Key Y: 0.0
			#Key Z: 0.113440313411918

			time.sleep(10)
			'''

			#	Get end effector to target location
			solveAnglesfromXYZ()

			time.sleep(10)

			rate.sleep()

	except rospy.ROSInterruptException as e: #https://answers.ros.org/question/244271/how-to-enter-the-callback-function-only-whenever-a-topic-is-updated-but-otherwise-keep-doing-something-else/
		print("Node has already been called:")
		print(e)
		pass