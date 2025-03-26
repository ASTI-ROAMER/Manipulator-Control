import numpy as np
import math
import time
import cv2
import rospy

import matplotlib.pyplot as plt
from scipy.integrate import odeint #ode64 matlab
from scipy.optimize import minimize #fmincon matlab
#matplotlib.rcParams['backend'] = 'TkAgg' #'QtAgg'

from std_msgs.msg import Float64
from sensor_msgs.msg import JointState

import time

import faulthandler
faulthandler.enable()

#ardu_testdata.txt

dataArduino = np.genfromtxt("ardu_sensor_TrajCheck.txt", dtype=float, 
                     encoding=None, delimiter=",") #q1, q2, q3, q4, iter, dT, Tau1, Tau2, Tau3, Tau4

q1_ardu = dataArduino[:,0]
q2_ardu = dataArduino[:,1]
q3_ardu = dataArduino[:,2]
q4_ardu = dataArduino[:,3]
iter_ardu = dataArduino[:,4]


dataDH = np.genfromtxt("ardu_sensor_DH.txt", dtype=float, 
                     encoding=None, delimiter=",") #q1, q2, q3, q4, iter, dT, Tau1, Tau2, Tau3, Tau4

q1_DH = dataDH[:,0]
q2_DH = dataDH[:,1]
q3_DH = dataDH[:,2]
q4_DH = dataDH[:,3]
iter_DH = dataDH[:,4]

dataServo = np.genfromtxt("ardu_sensor_Servo.txt", dtype=float, 
                     encoding=None, delimiter=",") #q1, q2, q3, q4, iter, dT, Tau1, Tau2, Tau3, Tau4

q1_Servo = dataServo[:,0]
q2_Servo = dataServo[:,1]
q3_Servo = dataServo[:,2]
q4_Servo = dataServo[:,3]
iter_Servo = dataServo[:,4]


# 1. Set trajectory duration/length to get to setpoint
tfinal = 1

# 2. Set number of iterations for splitting the tfinal (granularity)
Iterations = tfinal*100 #Used in plotting
TimeIncrement = 10**(-2) #Used in trajectory generation

# 3. Set time/x-axis for simulation
t = np.linspace(0, tfinal, Iterations)

# 4. Plot the stuff
fig, ax = plt.subplots(nrows=3, ncols=4)
x = np.linspace(0, tfinal, Iterations)

ax[0,0].plot(x, q1_ardu, label = "q_ardu")
ax[0,1].plot(x, q2_ardu, label = "q_ardu")
ax[0,2].plot(x, q3_ardu, label = "q_ardu")
ax[0,3].plot(x, q4_ardu, label = "q_ardu")

ax[1,0].plot(x, q1_DH, label = "q_DH")
ax[1,1].plot(x, q2_DH, label = "q_DH")
ax[1,2].plot(x, q3_DH, label = "q_DH")
ax[1,3].plot(x, q4_DH, label = "q_DH")

ax[2,0].plot(x, q1_Servo, label = "q_Servo")
ax[2,1].plot(x, q2_Servo, label = "q_Servo")
ax[2,2].plot(x, q3_Servo, label = "q_Servo")
ax[2,3].plot(x, q4_Servo, label = "q_Servo")


# plot 2 subplots
ax[0,0].set_title('q1')
ax[0,1].set_title('q2')
ax[0,2].set_title('q3')
ax[0,3].set_title('q4')

ax[1,0].set_title('q1_DH')
ax[1,1].set_title('q2_DH')
ax[1,2].set_title('q3_DH')
ax[1,3].set_title('q4_DH')

ax[2,0].set_title('q1_Servo')
ax[2,1].set_title('q2_Servo')
ax[2,2].set_title('q3_Servo')
ax[2,3].set_title('q4_Servo')


fig.suptitle('Joint movement of robot arm')
plt.show()
