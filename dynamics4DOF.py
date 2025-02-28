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


def DNK(d1, q1, q2, q3, q4, m1, m2, m3, m4, a2, a3, a4):
    #%%%%%%%%%%%%%%%%%%%%%% Dnk %%%%%%%%%%%%%%%%%%%%%%%%
    DNK_subs = np.zeros((4,4))

    
    #D11
    DNK_subs[0,0] = 0.08333333333333333 * (d1 ** 2) * m1 + 0.08333333333333333 * (a2 ** 2) * m2 * (np.cos(q2) ** 2) + 0.08333333333333333 * (a3 ** 2) * m3 * (np.cos(q2 + q3) ** 2) + 0.08333333333333333 * (a4 ** 2) * m4 * (np.cos(q2 + q3 + q4) ** 2) + m2 * (0.25 * (a2 ** 2) * (np.cos(q1) ** 2) * (np.cos(q2) ** 2) + 0.25 * (a2 ** 2) * (np.cos(q2) ** 2) * (np.sin(q1) ** 2)) + m3 * ((-0.5 * a3 * np.cos(q1) * np.cos(q2 + q3) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3))) ** 2 + (0.5 * a3 * np.cos(q2 + q3) * np.sin(q1) -((a2 * np.cos(q2) + a3 * np.cos(q2 + q3)) * np.sin(q1))) ** 2) + m4 * ((-0.5 * a4 * np.cos(q1) * np.cos(q2 + q3 + q4) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4))) ** 2 + (0.5 * a4 * np.cos(q2 + q3 + q4) * np.sin(q1) -((a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4)) * np.sin(q1))) ** 2)
    #D12
    DNK_subs[0,1] = m3 * (-((-0.5 * a3 * np.cos(q1) * np.cos(q2 + q3) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3))) * np.sin(q1) * (a2 * np.sin(q2) + 0.5 * a3 * np.sin(q2 + q3))) -(np.cos(q1) * (0.5 * a3 * np.cos(q2 + q3) * np.sin(q1) -((a2 * np.cos(q2) + a3 * np.cos(q2 + q3)) * np.sin(q1))) * (a2 * np.sin(q2) + 0.5 * a3 * np.sin(q2 + q3)))) + m4 * (-((-0.5 * a4 * np.cos(q1) * np.cos(q2 + q3 + q4) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4))) * np.sin(q1) * (a2 * np.sin(q2) + a3 * np.sin(q2 + q3) + 0.5 * a4 * np.sin(q2 + q3 + q4))) -(np.cos(q1) * (0.5 * a4 * np.cos(q2 + q3 + q4) * np.sin(q1) -((a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4)) * np.sin(q1))) * (a2 * np.sin(q2) + a3 * np.sin(q2 + q3) + 0.5 * a4 * np.sin(q2 + q3 + q4))))
    #D13
    DNK_subs[0,2] = m3 * (-0.5 * a3 * (-0.5 * a3 * np.cos(q1) * np.cos(q2 + q3) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3))) * np.sin(q1) * np.sin(q2 + q3) -0.5 * a3 * np.cos(q1) * (0.5 * a3 * np.cos(q2 + q3) * np.sin(q1) -((a2 * np.cos(q2) + a3 * np.cos(q2 + q3)) * np.sin(q1))) * np.sin(q2 + q3)) + m4 * (-((-0.5 * a4 * np.cos(q1) * np.cos(q2 + q3 + q4) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4))) * np.sin(q1) * (a3 * np.sin(q2 + q3) + 0.5 * a4 * np.sin(q2 + q3 + q4))) -(np.cos(q1) * (0.5 * a4 * np.cos(q2 + q3 + q4) * np.sin(q1) -((a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4)) * np.sin(q1))) * (a3 * np.sin(q2 + q3) + 0.5 * a4 * np.sin(q2 + q3 + q4))))
    #D14
    DNK_subs[0,3] = m4 * (-0.5 * a4 * (-0.5 * a4 * np.cos(q1) * np.cos(q2 + q3 + q4) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4))) * np.sin(q1) * np.sin(q2 + q3 + q4) -0.5 * a4 * np.cos(q1) * (0.5 * a4 * np.cos(q2 + q3 + q4) * np.sin(q1) -((a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4)) * np.sin(q1))) * np.sin(q2 + q3 + q4))

    #D21
    DNK_subs[1,0] = m3 * (-((-0.5 * a3 * np.cos(q1) * np.cos(q2 + q3) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3))) * np.sin(q1) * (a2 * np.sin(q2) + 0.5 * a3 * np.sin(q2 + q3))) -(np.cos(q1) * (0.5 * a3 * np.cos(q2 + q3) * np.sin(q1) -((a2 * np.cos(q2) + a3 * np.cos(q2 + q3)) * np.sin(q1))) * (a2 * np.sin(q2) + 0.5 * a3 * np.sin(q2 + q3)))) + m4 * (-((-0.5 * a4 * np.cos(q1) * np.cos(q2 + q3 + q4) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4))) * np.sin(q1) * (a2 * np.sin(q2) + a3 * np.sin(q2 + q3) + 0.5 * a4 * np.sin(q2 + q3 + q4))) -(np.cos(q1) * (0.5 * a4 * np.cos(q2 + q3 + q4) * np.sin(q1) -((a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4)) * np.sin(q1))) * (a2 * np.sin(q2) + a3 * np.sin(q2 + q3) + 0.5 * a4 * np.sin(q2 + q3 + q4))))
    #D22
    DNK_subs[1,1] = 0.08333333333333333 * (a2 ** 2) * m2 * (np.cos(q1) ** 2) * (np.cos(q1) ** 2 + np.sin(q1) ** 2) + 0.08333333333333333 * (a3 ** 2) * m3 * (np.cos(q1) ** 2) * (np.cos(q1) ** 2 + np.sin(q1) ** 2) + 0.08333333333333333 * (a4 ** 2) * m4 * (np.cos(q1) ** 2) * (np.cos(q1) ** 2 + np.sin(q1) ** 2) + 0.08333333333333333 * (a2 ** 2) * m2 * (np.sin(q1) ** 2) * (np.cos(q1) ** 2 + np.sin(q1) ** 2) + 0.08333333333333333 * (a3 ** 2) * m3 * (np.sin(q1) ** 2) * (np.cos(q1) ** 2 + np.sin(q1) ** 2) + 0.08333333333333333 * (a4 ** 2) * m4 * (np.sin(q1) ** 2) * (np.cos(q1) ** 2 + np.sin(q1) ** 2) + m2 * ((0.5 * a2 * (np.cos(q1) ** 2) * np.cos(q2) + 0.5 * a2 * np.cos(q2) * (np.sin(q1) ** 2)) ** 2 + 0.25 * (a2 ** 2) * (np.cos(q1) ** 2) * (np.sin(q2) ** 2) + 0.25 * (a2 ** 2) * (np.sin(q1) ** 2) * (np.sin(q2) ** 2)) + m3 * ((np.cos(q1) * (-0.5 * a3 * np.cos(q1) * np.cos(q2 + q3) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3))) + np.sin(q1) * (-0.5 * a3 * np.cos(q2 + q3) * np.sin(q1) + (a2 * np.cos(q2) + a3 * np.cos(q2 + q3)) * np.sin(q1))) ** 2 + (np.cos(q1) ** 2) * ((a2 * np.sin(q2) + 0.5 * a3 * np.sin(q2 + q3)) ** 2) + (np.sin(q1) ** 2) * ((a2 * np.sin(q2) + 0.5 * a3 * np.sin(q2 + q3)) ** 2)) + m4 * ((np.cos(q1) * (-0.5 * a4 * np.cos(q1) * np.cos(q2 + q3 + q4) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4))) + np.sin(q1) * (-0.5 * a4 * np.cos(q2 + q3 + q4) * np.sin(q1) + (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4)) * np.sin(q1))) ** 2 + (np.cos(q1) ** 2) * ((a2 * np.sin(q2) + a3 * np.sin(q2 + q3) + 0.5 * a4 * np.sin(q2 + q3 + q4)) ** 2) + (np.sin(q1) ** 2) * ((a2 * np.sin(q2) + a3 * np.sin(q2 + q3) + 0.5 * a4 * np.sin(q2 + q3 + q4)) ** 2))
    #D23
    DNK_subs[1,2] = 0.08333333333333333 * (a3 ** 2) * m3 * (np.cos(q1) ** 2) * (np.cos(q1) ** 2 + np.sin(q1) ** 2) + 0.08333333333333333 * (a4 ** 2) * m4 * (np.cos(q1) ** 2) * (np.cos(q1) ** 2 + np.sin(q1) ** 2) + 0.08333333333333333 * (a3 ** 2) * m3 * (np.sin(q1) ** 2) * (np.cos(q1) ** 2 + np.sin(q1) ** 2) + 0.08333333333333333 * (a4 ** 2) * m4 * (np.sin(q1) ** 2) * (np.cos(q1) ** 2 + np.sin(q1) ** 2) + m3 * ((np.cos(q1) * (-0.5 * a3 * np.cos(q1) * np.cos(q2 + q3) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3))) + np.sin(q1) * (-0.5 * a3 * np.cos(q2 + q3) * np.sin(q1) + (a2 * np.cos(q2) + a3 * np.cos(q2 + q3)) * np.sin(q1))) * (np.cos(q1) * (-(a2 * np.cos(q1) * np.cos(q2)) -0.5 * a3 * np.cos(q1) * np.cos(q2 + q3) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3))) + np.sin(q1) * (-(a2 * np.cos(q2) * np.sin(q1)) -0.5 * a3 * np.cos(q2 + q3) * np.sin(q1) + (a2 * np.cos(q2) + a3 * np.cos(q2 + q3)) * np.sin(q1))) + 0.5 * a3 * (np.cos(q1) ** 2) * np.sin(q2 + q3) * (a2 * np.sin(q2) + 0.5 * a3 * np.sin(q2 + q3)) + 0.5 * a3 * (np.sin(q1) ** 2) * np.sin(q2 + q3) * (a2 * np.sin(q2) + 0.5 * a3 * np.sin(q2 + q3))) + m4 * ((np.cos(q1) * (-0.5 * a4 * np.cos(q1) * np.cos(q2 + q3 + q4) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4))) + np.sin(q1) * (-0.5 * a4 * np.cos(q2 + q3 + q4) * np.sin(q1) + (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4)) * np.sin(q1))) * (np.cos(q1) * (-(a2 * np.cos(q1) * np.cos(q2)) -0.5 * a4 * np.cos(q1) * np.cos(q2 + q3 + q4) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4))) + np.sin(q1) * (-(a2 * np.cos(q2) * np.sin(q1)) -0.5 * a4 * np.cos(q2 + q3 + q4) * np.sin(q1) + (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4)) * np.sin(q1))) + (np.cos(q1) ** 2) * (a3 * np.sin(q2 + q3) + 0.5 * a4 * np.sin(q2 + q3 + q4)) * (a2 * np.sin(q2) + a3 * np.sin(q2 + q3) + 0.5 * a4 * np.sin(q2 + q3 + q4)) + (np.sin(q1) ** 2) * (a3 * np.sin(q2 + q3) + 0.5 * a4 * np.sin(q2 + q3 + q4)) * (a2 * np.sin(q2) + a3 * np.sin(q2 + q3) + 0.5 * a4 * np.sin(q2 + q3 + q4)))
    #D24
    DNK_subs[1,3] = 0.08333333333333333 * (a4 ** 2) * m4 * (np.cos(q1) ** 2) * (np.cos(q1) ** 2 + np.sin(q1) ** 2) + 0.08333333333333333 * (a4 ** 2) * m4 * (np.sin(q1) ** 2) * (np.cos(q1) ** 2 + np.sin(q1) ** 2) + m4 * ((np.cos(q1) * (-0.5 * a4 * np.cos(q1) * np.cos(q2 + q3 + q4) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4))) + np.sin(q1) * (-0.5 * a4 * np.cos(q2 + q3 + q4) * np.sin(q1) + (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4)) * np.sin(q1))) * (np.cos(q1) * (-(np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3))) -0.5 * a4 * np.cos(q1) * np.cos(q2 + q3 + q4) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4))) + np.sin(q1) * (-((a2 * np.cos(q2) + a3 * np.cos(q2 + q3)) * np.sin(q1)) -0.5 * a4 * np.cos(q2 + q3 + q4) * np.sin(q1) + (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4)) * np.sin(q1))) + 0.5 * a4 * (np.cos(q1) ** 2) * np.sin(q2 + q3 + q4) * (a2 * np.sin(q2) + a3 * np.sin(q2 + q3) + 0.5 * a4 * np.sin(q2 + q3 + q4)) + 0.5 * a4 * (np.sin(q1) ** 2) * np.sin(q2 + q3 + q4) * (a2 * np.sin(q2) + a3 * np.sin(q2 + q3) + 0.5 * a4 * np.sin(q2 + q3 + q4)))
 
    #D31
    DNK_subs[2,0] = m3 * (-0.5 * a3 * (-0.5 * a3 * np.cos(q1) * np.cos(q2 + q3) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3))) * np.sin(q1) * np.sin(q2 + q3) -0.5 * a3 * np.cos(q1) * (0.5 * a3 * np.cos(q2 + q3) * np.sin(q1) -((a2 * np.cos(q2) + a3 * np.cos(q2 + q3)) * np.sin(q1))) * np.sin(q2 + q3)) + m4 * (-((-0.5 * a4 * np.cos(q1) * np.cos(q2 + q3 + q4) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4))) * np.sin(q1) * (a3 * np.sin(q2 + q3) + 0.5 * a4 * np.sin(q2 + q3 + q4))) -(np.cos(q1) * (0.5 * a4 * np.cos(q2 + q3 + q4) * np.sin(q1) -((a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4)) * np.sin(q1))) * (a3 * np.sin(q2 + q3) + 0.5 * a4 * np.sin(q2 + q3 + q4))))
    #D32
    DNK_subs[2,1] = 0.08333333333333333 * (a3 ** 2) * m3 * (np.cos(q1) ** 2) * (np.cos(q1) ** 2 + np.sin(q1) ** 2) + 0.08333333333333333 * (a4 ** 2) * m4 * (np.cos(q1) ** 2) * (np.cos(q1) ** 2 + np.sin(q1) ** 2) + 0.08333333333333333 * (a3 ** 2) * m3 * (np.sin(q1) ** 2) * (np.cos(q1) ** 2 + np.sin(q1) ** 2) + 0.08333333333333333 * (a4 ** 2) * m4 * (np.sin(q1) ** 2) * (np.cos(q1) ** 2 + np.sin(q1) ** 2) + m3 * ((np.cos(q1) * (-0.5 * a3 * np.cos(q1) * np.cos(q2 + q3) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3))) + np.sin(q1) * (-0.5 * a3 * np.cos(q2 + q3) * np.sin(q1) + (a2 * np.cos(q2) + a3 * np.cos(q2 + q3)) * np.sin(q1))) * (np.cos(q1) * (-(a2 * np.cos(q1) * np.cos(q2)) -0.5 * a3 * np.cos(q1) * np.cos(q2 + q3) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3))) + np.sin(q1) * (-(a2 * np.cos(q2) * np.sin(q1)) -0.5 * a3 * np.cos(q2 + q3) * np.sin(q1) + (a2 * np.cos(q2) + a3 * np.cos(q2 + q3)) * np.sin(q1))) + 0.5 * a3 * (np.cos(q1) ** 2) * np.sin(q2 + q3) * (a2 * np.sin(q2) + 0.5 * a3 * np.sin(q2 + q3)) + 0.5 * a3 * (np.sin(q1) ** 2) * np.sin(q2 + q3) * (a2 * np.sin(q2) + 0.5 * a3 * np.sin(q2 + q3))) + m4 * ((np.cos(q1) * (-0.5 * a4 * np.cos(q1) * np.cos(q2 + q3 + q4) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4))) + np.sin(q1) * (-0.5 * a4 * np.cos(q2 + q3 + q4) * np.sin(q1) + (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4)) * np.sin(q1))) * (np.cos(q1) * (-(a2 * np.cos(q1) * np.cos(q2)) -0.5 * a4 * np.cos(q1) * np.cos(q2 + q3 + q4) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4))) + np.sin(q1) * (-(a2 * np.cos(q2) * np.sin(q1)) -0.5 * a4 * np.cos(q2 + q3 + q4) * np.sin(q1) + (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4)) * np.sin(q1))) + (np.cos(q1) ** 2) * (a3 * np.sin(q2 + q3) + 0.5 * a4 * np.sin(q2 + q3 + q4)) * (a2 * np.sin(q2) + a3 * np.sin(q2 + q3) + 0.5 * a4 * np.sin(q2 + q3 + q4)) + (np.sin(q1) ** 2) * (a3 * np.sin(q2 + q3) + 0.5 * a4 * np.sin(q2 + q3 + q4)) * (a2 * np.sin(q2) + a3 * np.sin(q2 + q3) + 0.5 * a4 * np.sin(q2 + q3 + q4)))
    #D33
    DNK_subs[2,2] = 0.08333333333333333 * (a3 ** 2) * m3 * (np.cos(q1) ** 2) * (np.cos(q1) ** 2 + np.sin(q1) ** 2) + 0.08333333333333333 * (a4 ** 2) * m4 * (np.cos(q1) ** 2) * (np.cos(q1) ** 2 + np.sin(q1) ** 2) + 0.08333333333333333 * (a3 ** 2) * m3 * (np.sin(q1) ** 2) * (np.cos(q1) ** 2 + np.sin(q1) ** 2) + 0.08333333333333333 * (a4 ** 2) * m4 * (np.sin(q1) ** 2) * (np.cos(q1) ** 2 + np.sin(q1) ** 2) + m3 * ((np.cos(q1) * (-(a2 * np.cos(q1) * np.cos(q2)) -0.5 * a3 * np.cos(q1) * np.cos(q2 + q3) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3))) + np.sin(q1) * (-(a2 * np.cos(q2) * np.sin(q1)) -0.5 * a3 * np.cos(q2 + q3) * np.sin(q1) + (a2 * np.cos(q2) + a3 * np.cos(q2 + q3)) * np.sin(q1))) ** 2 + 0.25 * (a3 ** 2) * (np.cos(q1) ** 2) * (np.sin(q2 + q3) ** 2) + 0.25 * (a3 ** 2) * (np.sin(q1) ** 2) * (np.sin(q2 + q3) ** 2)) + m4 * ((np.cos(q1) * (-(a2 * np.cos(q1) * np.cos(q2)) -0.5 * a4 * np.cos(q1) * np.cos(q2 + q3 + q4) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4))) + np.sin(q1) * (-(a2 * np.cos(q2) * np.sin(q1)) -0.5 * a4 * np.cos(q2 + q3 + q4) * np.sin(q1) + (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4)) * np.sin(q1))) ** 2 + (np.cos(q1) ** 2) * ((a3 * np.sin(q2 + q3) + 0.5 * a4 * np.sin(q2 + q3 + q4)) ** 2) + (np.sin(q1) ** 2) * ((a3 * np.sin(q2 + q3) + 0.5 * a4 * np.sin(q2 + q3 + q4)) ** 2))
    #D34
    DNK_subs[2,3] = 0.08333333333333333 * (a4 ** 2) * m4 * (np.cos(q1) ** 2) * (np.cos(q1) ** 2 + np.sin(q1) ** 2) + 0.08333333333333333 * (a4 ** 2) * m4 * (np.sin(q1) ** 2) * (np.cos(q1) ** 2 + np.sin(q1) ** 2) + m4 * ((np.cos(q1) * (-(a2 * np.cos(q1) * np.cos(q2)) -0.5 * a4 * np.cos(q1) * np.cos(q2 + q3 + q4) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4))) + np.sin(q1) * (-(a2 * np.cos(q2) * np.sin(q1)) -0.5 * a4 * np.cos(q2 + q3 + q4) * np.sin(q1) + (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4)) * np.sin(q1))) * (np.cos(q1) * (-(np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3))) -0.5 * a4 * np.cos(q1) * np.cos(q2 + q3 + q4) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4))) + np.sin(q1) * (-((a2 * np.cos(q2) + a3 * np.cos(q2 + q3)) * np.sin(q1)) -0.5 * a4 * np.cos(q2 + q3 + q4) * np.sin(q1) + (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4)) * np.sin(q1))) + 0.5 * a4 * (np.cos(q1) ** 2) * np.sin(q2 + q3 + q4) * (a3 * np.sin(q2 + q3) + 0.5 * a4 * np.sin(q2 + q3 + q4)) + 0.5 * a4 * (np.sin(q1) ** 2) * np.sin(q2 + q3 + q4) * (a3 * np.sin(q2 + q3) + 0.5 * a4 * np.sin(q2 + q3 + q4)))
 
    #D41
    DNK_subs[3,0] = m4 * (-0.5 * a4 * (-0.5 * a4 * np.cos(q1) * np.cos(q2 + q3 + q4) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4))) * np.sin(q1) * np.sin(q2 + q3 + q4) -0.5 * a4 * np.cos(q1) * (0.5 * a4 * np.cos(q2 + q3 + q4) * np.sin(q1) -((a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4)) * np.sin(q1))) * np.sin(q2 + q3 + q4))
    #D42
    DNK_subs[3,1] = 0.08333333333333333 * (a4 ** 2) * m4 * (np.cos(q1) ** 2) * (np.cos(q1) ** 2 + np.sin(q1) ** 2) + 0.08333333333333333 * (a4 ** 2) * m4 * (np.sin(q1) ** 2) * (np.cos(q1) ** 2 + np.sin(q1) ** 2) + m4 * ((np.cos(q1) * (-0.5 * a4 * np.cos(q1) * np.cos(q2 + q3 + q4) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4))) + np.sin(q1) * (-0.5 * a4 * np.cos(q2 + q3 + q4) * np.sin(q1) + (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4)) * np.sin(q1))) * (np.cos(q1) * (-(np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3))) -0.5 * a4 * np.cos(q1) * np.cos(q2 + q3 + q4) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4))) + np.sin(q1) * (-((a2 * np.cos(q2) + a3 * np.cos(q2 + q3)) * np.sin(q1)) -0.5 * a4 * np.cos(q2 + q3 + q4) * np.sin(q1) + (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4)) * np.sin(q1))) + 0.5 * a4 * (np.cos(q1) ** 2) * np.sin(q2 + q3 + q4) * (a2 * np.sin(q2) + a3 * np.sin(q2 + q3) + 0.5 * a4 * np.sin(q2 + q3 + q4)) + 0.5 * a4 * (np.sin(q1) ** 2) * np.sin(q2 + q3 + q4) * (a2 * np.sin(q2) + a3 * np.sin(q2 + q3) + 0.5 * a4 * np.sin(q2 + q3 + q4)))
    #D43
    DNK_subs[3,2] = 0.08333333333333333 * (a4 ** 2) * m4 * (np.cos(q1) ** 2) * (np.cos(q1) ** 2 + np.sin(q1) ** 2) + 0.08333333333333333 * (a4 ** 2) * m4 * (np.sin(q1) ** 2) * (np.cos(q1) ** 2 + np.sin(q1) ** 2) + m4 * ((np.cos(q1) * (-(a2 * np.cos(q1) * np.cos(q2)) -0.5 * a4 * np.cos(q1) * np.cos(q2 + q3 + q4) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4))) + np.sin(q1) * (-(a2 * np.cos(q2) * np.sin(q1)) -0.5 * a4 * np.cos(q2 + q3 + q4) * np.sin(q1) + (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4)) * np.sin(q1))) * (np.cos(q1) * (-(np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3))) -0.5 * a4 * np.cos(q1) * np.cos(q2 + q3 + q4) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4))) + np.sin(q1) * (-((a2 * np.cos(q2) + a3 * np.cos(q2 + q3)) * np.sin(q1)) -0.5 * a4 * np.cos(q2 + q3 + q4) * np.sin(q1) + (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4)) * np.sin(q1))) + 0.5 * a4 * (np.cos(q1) ** 2) * np.sin(q2 + q3 + q4) * (a3 * np.sin(q2 + q3) + 0.5 * a4 * np.sin(q2 + q3 + q4)) + 0.5 * a4 * (np.sin(q1) ** 2) * np.sin(q2 + q3 + q4) * (a3 * np.sin(q2 + q3) + 0.5 * a4 * np.sin(q2 + q3 + q4)))
    #D44
    DNK_subs[3,3] = 0.08333333333333333 * (a4 ** 2) * m4 * (np.cos(q1) ** 2) * (np.cos(q1) ** 2 + np.sin(q1) ** 2) + 0.08333333333333333 * (a4 ** 2) * m4 * (np.sin(q1) ** 2) * (np.cos(q1) ** 2 + np.sin(q1) ** 2) + m4 * ((np.cos(q1) * (-(np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3))) -0.5 * a4 * np.cos(q1) * np.cos(q2 + q3 + q4) + np.cos(q1) * (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4))) + np.sin(q1) * (-((a2 * np.cos(q2) + a3 * np.cos(q2 + q3)) * np.sin(q1)) -0.5 * a4 * np.cos(q2 + q3 + q4) * np.sin(q1) + (a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4)) * np.sin(q1))) ** 2 + 0.25 * (a4 ** 2) * (np.cos(q1) ** 2) * (np.sin(q2 + q3 + q4) ** 2) + 0.25 * (a4 ** 2) * (np.sin(q1) ** 2) * (np.sin(q2 + q3 + q4) ** 2))
    return DNK_subs

def DNKJ(d1, q1, q2, q3, q4, m1, m2, m3, m4, a2, a3, a4):
    #%%%%%%%%%%%%%%%%%%%%%% DNKJ %%%%%%%%%%%%%%%%%%%%%%%%
    DNKJ_subs = np.zeros((4,4,4))

    #D[1,1,1]
    DNKJ_subs[0,0,0] = 0
    #D[1,1,2]
    DNKJ_subs[0,0,1] = 0.1666666666666667 * (((a2 ** 2) * m2 + 3 * (a2 ** 2) * m3 + 3 * (a2 ** 2) * m4) * np.sin(2 * q2) + (3 * a2 * a3 * m3 + 6 * a2 * a3 * m4) * np.sin(2 * q2 + q3) + ((a3 ** 2) * m3 + 3 * (a3 ** 2) * m4) * np.sin(2 * q2 + 2 * q3) + 3 * a2 * a4 * m4 * np.sin(2 * q2 + q3 + q4) + 3 * a3 * a4 * m4 * np.sin(2 * q2 + 2 * q3 + q4) + (a4 ** 2) * m4 * np.sin(2 * q2 + 2 * q3 + 2 * q4))
    #D[1,1,3]
    DNKJ_subs[0,0,2] = 0.08333333333333333 * ((3 * a2 * a3 * m3 + 6 * a2 * a3 * m4) * np.sin(q3) + (3 * a2 * a3 * m3 + 6 * a2 * a3 * m4) * np.sin(2 * q2 + q3) + (2 * (a3 ** 2) * m3 + 6 * (a3 ** 2) * m4) * np.sin(2 * q2 + 2 * q3) + 3 * a2 * a4 * m4 * np.sin(q3 + q4) + 3 * a2 * a4 * m4 * np.sin(2 * q2 + q3 + q4) + 6 * a3 * a4 * m4 * np.sin(2 * q2 + 2 * q3 + q4) + 2 * (a4 ** 2) * m4 * np.sin(2 * q2 + 2 * q3 + 2 * q4))
    #D[1,1,4]
    DNKJ_subs[0,0,3] = 0.1666666666666667 * a4 * m4 * (3 * a2 * np.cos(q2) + 3 * a3 * np.cos(q2 + q3) + 2 * a4 * np.cos(q2 + q3 + q4)) * np.sin(q2 + q3 + q4)

    #D[1,2,1]
    DNKJ_subs[0,1,0] = 0.1666666666666667 * ((-((a2 ** 2) * m2) -3 * (a2 ** 2) * m3 -3 * (a2 ** 2) * m4) * np.sin(2 * q2) + (-3 * a2 * a3 * m3 -6 * a2 * a3 * m4) * np.sin(2 * q2 + q3) + (-((a3 ** 2) * m3) -3 * (a3 ** 2) * m4) * np.sin(2 * q2 + 2 * q3) -3 * a2 * a4 * m4 * np.sin(2 * q2 + q3 + q4) -3 * a3 * a4 * m4 * np.sin(2 * q2 + 2 * q3 + q4) -((a4 ** 2) * m4 * np.sin(2 * q2 + 2 * q3 + 2 * q4)))
    #D[1,2,2]
    DNKJ_subs[0,1,1] = 0
    #D[1,2,3]
    DNKJ_subs[0,1,2] = 0
    #D[1,2,4]
    DNKJ_subs[0,1,3] = 0

    #D[1,3,1]
    DNKJ_subs[0,2,0] = 0.08333333333333333 * ((-3 * a2 * a3 * m3 -6 * a2 * a3 * m4) * np.sin(q3) + (-3 * a2 * a3 * m3 -6 * a2 * a3 * m4) * np.sin(2 * q2 + q3) + (-2 * (a3 ** 2) * m3 -6 * (a3 ** 2) * m4) * np.sin(2 * q2 + 2 * q3) -3 * a2 * a4 * m4 * np.sin(q3 + q4) -3 * a2 * a4 * m4 * np.sin(2 * q2 + q3 + q4) -6 * a3 * a4 * m4 * np.sin(2 * q2 + 2 * q3 + q4) -2 * (a4 ** 2) * m4 * np.sin(2 * q2 + 2 * q3 + 2 * q4))
    #D[1,3,2]
    DNKJ_subs[0,2,1] = 0
    #D[1,3,3]
    DNKJ_subs[0,2,2] = 0
    #D[1,3,4]
    DNKJ_subs[0,2,3] = 0

    #D[1,4,1]
    DNKJ_subs[0,3,0] = -0.1666666666666667 * a4 * m4 * (3 * a2 * np.cos(q2) + 3 * a3 * np.cos(q2 + q3) + 2 * a4 * np.cos(q2 + q3 + q4)) * np.sin(q2 + q3 + q4)
    #D[1,4,2]
    DNKJ_subs[0,3,1] = 0
    #D[1,4,3]
    DNKJ_subs[0,3,2] = 0
    #D[1,4,4]
    DNKJ_subs[0,3,3] = 0

    #--

    #D[2,1,1]
    DNKJ_subs[1,0,0] = 0.1666666666666667 * ((-((a2 ** 2) * m2) -3 * (a2 ** 2) * m3 -3 * (a2 ** 2) * m4) * np.sin(2 * q2) + (-3 * a2 * a3 * m3 -6 * a2 * a3 * m4) * np.sin(2 * q2 + q3) + (-((a3 ** 2) * m3) -3 * (a3 ** 2) * m4) * np.sin(2 * q2 + 2 * q3) -3 * a2 * a4 * m4 * np.sin(2 * q2 + q3 + q4) -3 * a3 * a4 * m4 * np.sin(2 * q2 + 2 * q3 + q4) -((a4 ** 2) * m4 * np.sin(2 * q2 + 2 * q3 + 2 * q4)))
    #D[2,1,2]
    DNKJ_subs[1,0,1] = 0
    #D[2,1,3]
    DNKJ_subs[1,0,2] = 0
    #D[2,1,4]
    DNKJ_subs[1,0,3] = 0

    #D[2,2,1]
    DNKJ_subs[1,1,0] = 0
    #D[2,2,2]
    DNKJ_subs[1,1,1] = 0
    #D[2,2,3]
    DNKJ_subs[1,1,2] = 0.5 * a2 * ((a3 * m3 + 2 * a3 * m4) * np.sin(q3) + a4 * m4 * np.sin(q3 + q4))
    #D[2,2,4]
    DNKJ_subs[1,1,3] = 0.5 * a4 * m4 * (a3 * np.sin(q4) + a2 * np.sin(q3 + q4))

    #D[2,3,1]
    DNKJ_subs[1,2,0] = 0
    #D[2,3,2]
    DNKJ_subs[1,2,1] = -0.5 * a2 * ((a3 * m3 + 2 * a3 * m4) * np.sin(q3) + a4 * m4 * np.sin(q3 + q4))
    #D[2,3,3]
    DNKJ_subs[1,2,2] = 0
    #D[2,3,4]
    DNKJ_subs[1,2,3] = a3 * a4 * m4 * np.cos(0.5 * q4) * np.sin(0.5 * q4)

    #D[2,4,1]
    DNKJ_subs[1,3,0] = 0
    #D[2,4,2]
    DNKJ_subs[1,3,1] = -0.5 * a4 * m4 * (a3 * np.sin(q4) + a2 * np.sin(q3 + q4))
    #D[2,4,3]
    DNKJ_subs[1,3,2] = -(a3 * a4 * m4 * np.cos(0.5 * q4) * np.sin(0.5 * q4))
    #D[2,4,4]
    DNKJ_subs[1,3,3] = 0

    #--

    #D[3,1,1]
    DNKJ_subs[2,0,0] = 0.08333333333333333 * ((-3 * a2 * a3 * m3 -6 * a2 * a3 * m4) * np.sin(q3) + (-3 * a2 * a3 * m3 -6 * a2 * a3 * m4) * np.sin(2 * q2 + q3) + (-2 * (a3 ** 2) * m3 -6 * (a3 ** 2) * m4) * np.sin(2 * q2 + 2 * q3) -3 * a2 * a4 * m4 * np.sin(q3 + q4) -3 * a2 * a4 * m4 * np.sin(2 * q2 + q3 + q4) -6 * a3 * a4 * m4 * np.sin(2 * q2 + 2 * q3 + q4) -2 * (a4 ** 2) * m4 * np.sin(2 * q2 + 2 * q3 + 2 * q4))
    #D[3,1,2]
    DNKJ_subs[2,0,1] = 0
    #D[3,1,3]
    DNKJ_subs[2,0,2] = 0
    #D[3,1,4]
    DNKJ_subs[2,0,3] = 0

    #D[3,2,1]
    DNKJ_subs[2,1,0] = 0
    #D[3,2,2]
    DNKJ_subs[2,1,1] = -0.5 * a2 * ((a3 * m3 + 2 * a3 * m4) * np.sin(q3) + a4 * m4 * np.sin(q3 + q4))
    #D[3,2,3]
    DNKJ_subs[2,1,2] = 0
    #D[3,2,4]
    DNKJ_subs[2,1,3] = a3 * a4 * m4 * np.cos(0.5 * q4) * np.sin(0.5 * q4)

    #D[3,3,1]
    DNKJ_subs[2,2,0] = 0
    #D[3,3,2]
    DNKJ_subs[2,2,1] = -0.5 * a2 * ((a3 * m3 + 2 * a3 * m4) * np.sin(q3) + a4 * m4 * np.sin(q3 + q4))
    #D[3,3,3]
    DNKJ_subs[2,2,2] = 0
    #D[3,3,4]
    DNKJ_subs[2,2,3] = a3 * a4 * m4 * np.cos(0.5 * q4) * np.sin(0.5 * q4)

    #D[3,4,1]
    DNKJ_subs[2,3,0] = 0
    #D[3,4,2]
    DNKJ_subs[2,3,1] = -0.5 * a4 * m4 * (a3 * np.sin(q4) + a2 * np.sin(q3 + q4))
    #D[3,4,3]
    DNKJ_subs[2,3,2] = -(a3 * a4 * m4 * np.cos(0.5 * q4) * np.sin(0.5 * q4))
    #D[3,4,4]
    DNKJ_subs[2,3,3] = 0

    #--

    #D[4,1,1]
    DNKJ_subs[3,0,0] = -0.1666666666666667 * a4 * m4 * (3 * a2 * np.cos(q2) + 3 * a3 * np.cos(q2 + q3) + 2 * a4 * np.cos(q2 + q3 + q4)) * np.sin(q2 + q3 + q4)
    #D[4,1,2]
    DNKJ_subs[3,0,1] = 0
    #D[4,1,3]
    DNKJ_subs[3,0,2] = 0
    #D[4,1,4]
    DNKJ_subs[3,0,3] = 0

    #D[4,2,1]
    DNKJ_subs[3,1,0] = 0
    #D[4,2,2]
    DNKJ_subs[3,1,1] = -0.5 * a4 * m4 * (a3 * np.sin(q4) + a2 * np.sin(q3 + q4))
    #D[4,2,3]
    DNKJ_subs[3,1,2] = -(a3 * a4 * m4 * np.cos(0.5 * q4) * np.sin(0.5 * q4))
    #D[4,2,4]
    DNKJ_subs[3,1,3] = 0

    #D[4,3,1]
    DNKJ_subs[3,2,0] = 0
    #D[4,3,2]
    DNKJ_subs[3,2,1] = -0.5 * a4 * m4 * (a3 * np.sin(q4) + a2 * np.sin(q3 + q4))
    #D[4,3,3]
    DNKJ_subs[3,2,2] = -(a3 * a4 * m4 * np.cos(0.5 * q4) * np.sin(0.5 * q4))
    #D[4,3,4]
    DNKJ_subs[3,2,3] = 0

    #D[4,4,1]
    DNKJ_subs[3,3,0] = 0
    #D[4,4,2]
    DNKJ_subs[3,3,1] = -0.5 * a4 * m4 * (a3 * np.sin(q4) + a2 * np.sin(q3 + q4))
    #D[4,4,3]
    DNKJ_subs[3,3,2] = -(a3 * a4 * m4 * np.cos(0.5 * q4) * np.sin(0.5 * q4))
    #D[4,4,4]
    DNKJ_subs[3,3,3] = 0


    return DNKJ_subs

def GN(d1, q1, q2, q3, q4, m1, m2, m3, m4, a2, a3, a4):
    #%%%%%%%%%%%%%%%%%%%%%% GN %%%%%%%%%%%%%%%%%%%%%%%%
    g=9.18;
    GN_subs = np.transpose(np.zeros(4)[np.newaxis])

    GN_subs[0] = 0
    GN_subs[1] = 0.5 * g * ((a2 * m2 + 2 * a2 * m3 + 2 * a2 * m4) * np.cos(q2) + (a3 * m3 + 2 * a3 * m4) * np.cos(q2 + q3) + a4 * m4 * np.cos(q2 + q3 + q4))
    GN_subs[2] = 0.5 * g * ((a3 * m3 + 2 * a3 * m4) * np.cos(q2 + q3) + a4 * m4 * np.cos(q2 + q3 + q4))
    GN_subs[3] = 0.5 * a4 * g * m4 * np.cos(q2 + q3 + q4)


    return GN_subs

def xdotTeleNoPID(x, t):
    #Albert Paolo P. Bugayong
    #global DA 
    global Tau
    #global G 
    #global DB

    l1 = 0.180
    d1 = l1
    l2 = 0.300
    l3 = 0.305
    l4 = 0.071

    m1 = 0.250
    m2 = 0.134
    m3 = 0.127
    m4 = 0.087


    q1 = x[0]
    q2 = x[1]
    q3 = x[2]
    q4 = x[3]

    Dnk = DNK(d1, q1, q2, q3, q4, m1, m2, m3, m4, l2, l3, l4)
    Dnkj = DNKJ(d1, q1, q2, q3, q4, m1, m2, m3, m4, l2, l3, l4)
    Gn = GN(d1, q1, q2, q3, q4, m1, m2, m3, m4, l2, l3, l4)

    DA = np.array([[Dnk[0,0], Dnk[0,1], Dnk[0,2], Dnk[0,3]],
     [Dnk[1,0], Dnk[1,1], Dnk[1,2], Dnk[1,3]],
     [Dnk[2,0], Dnk[2,1], Dnk[2,2], Dnk[2,3]],
     [Dnk[3,0], Dnk[3,1], Dnk[3,2], Dnk[3,3]]])

    DB = np.array([[Dnkj[0,0,0], Dnkj[0,0,1], Dnkj[0,0,2], Dnkj[0,0,3], Dnkj[0,1,0], Dnkj[0,1,1], Dnkj[0,1,2], Dnkj[0,1,3], Dnkj[0,2,0], Dnkj[0,2,1], Dnkj[0,2,2], Dnkj[0,2,3], Dnkj[0,3,0], Dnkj[0,3,1], Dnkj[0,3,2], Dnkj[0,3,3]],
      [Dnkj[1,0,0], Dnkj[1,0,1], Dnkj[1,0,2], Dnkj[1,0,3], Dnkj[1,1,0], Dnkj[1,1,1], Dnkj[1,1,2], Dnkj[1,1,3], Dnkj[1,2,0], Dnkj[1,2,1], Dnkj[1,2,2], Dnkj[1,2,3], Dnkj[1,3,0], Dnkj[1,3,1], Dnkj[1,3,2], Dnkj[1,3,3]],
      [Dnkj[2,0,0], Dnkj[2,0,1], Dnkj[2,0,2], Dnkj[2,0,3], Dnkj[2,1,0], Dnkj[2,1,1], Dnkj[2,1,2], Dnkj[2,1,3], Dnkj[2,2,0], Dnkj[2,2,1], Dnkj[2,2,2], Dnkj[2,2,3], Dnkj[2,3,0], Dnkj[2,3,1], Dnkj[2,3,2], Dnkj[2,3,3]],
      [Dnkj[3,0,0], Dnkj[3,0,1], Dnkj[3,0,2], Dnkj[3,0,3], Dnkj[3,1,0], Dnkj[3,1,1], Dnkj[3,1,2], Dnkj[3,1,3], Dnkj[3,2,0], Dnkj[3,2,1], Dnkj[3,2,2], Dnkj[3,2,3], Dnkj[3,3,0], Dnkj[3,3,1], Dnkj[3,3,2], Dnkj[3,3,3]]])

    G = np.array(Gn)


    Bf1 = 0 #1*G1
    Bf2 = 0 #0.5*G2
    Bf3 = 0
    Bf4 = 0
    Bm = [Bf1*x[4],  Bf2*x[5],  Bf3*x[6],  Bf4*x[7]]

    qdotdot = np.zeros((4,1))

    qdot = np.transpose(np.array([x[4]*x[4],x[4]*x[5],x[4]*x[6],x[4]*x[7],x[5]*x[4],x[5]*x[5],x[5]*x[6],x[5]*x[7],x[6]*x[4],x[6]*x[5],x[6]*x[6],x[6]*x[7],x[7]*x[4],x[7]*x[5],x[7]*x[6],x[7]*x[7]])[np.newaxis]) #x(6) = qdot1, x(1)= q1
    
    '''
    print('xdot')
    print('qdot') #(25, 1)
    print(qdot.shape)
    print('G')
    print(G.shape) #(5, 1)
    print('DA')
    print(DA.shape) #(5, 5)
    print('DB')
    print(DB.shape) #(5, 25)
    '''

    #qdotdot = inv(DA)*(Tau-Bm-G'-DB*qdot)
    qdotdot = np.matmul(np.linalg.inv(DA),(Tau-G-np.matmul(DB,qdot)))

    xdotTele_out = np.zeros(8)

    xdotTele_out[0] = x[4]
    xdotTele_out[1] = x[5]
    xdotTele_out[2] = x[6]
    xdotTele_out[3] = x[7]
    xdotTele_out[4] = qdotdot[0,0]
    xdotTele_out[5] = qdotdot[1,0]
    xdotTele_out[6] = qdotdot[2,0]
    xdotTele_out[7] = qdotdot[3,0]

    '''
    if xdotTele_out[3] < -1.57
    xdotTele_out[3] = -1.57
    else xdotTele_out[3] < 1.57
    xdotTele_out[3] = 1.57
    end
    '''
    return np.array(xdotTele_out)




# 1. Set trajectory duration/length to get to setpoint
tfinal = 1

# 2. Set number of iterations for splitting the tfinal (granularity)
Iterations = tfinal*100 #Used in plotting
TimeIncrement = 10**(-2) #Used in trajectory generation

# 3. Set time/x-axis for simulation
t = np.linspace(0, tfinal, Iterations)

# 4. Declare container for state space equation outputs
y0 = np.zeros(8) #[q1 q2 q3 q4 q5 q1dot q2dot q3dot q4dot q5dot]

# 5. Declare variables for plotting
y0_store = np.zeros((Iterations, 8))

Tau1_store = np.zeros((Iterations, 1))
Tau2_store = np.zeros((Iterations, 1))
Tau3_store = np.zeros((Iterations, 1))
Tau4_store = np.zeros((Iterations, 1))

err1_store = np.zeros((Iterations, 1))
err2_store = np.zeros((Iterations, 1))
err3_store = np.zeros((Iterations, 1))
err4_store = np.zeros((Iterations, 1))

q1_store = np.zeros((Iterations, 1))
q2_store = np.zeros((Iterations, 1))
q3_store = np.zeros((Iterations, 1))
q4_store = np.zeros((Iterations, 1))

# 6. Declare Torque Variables
Tau1 = 0
Tau2 = 0
Tau3 = 0
Tau4 = 0

# 7. Declare Error Terms
err1 = 0
err2 = 0
err3 = 0
err4 = 0

errsum1 = 0
errsum2 = 0
errsum3 = 0
errsum4 = 0

# 8. Declare PID Gains



Kp_term1 = 100
Kp_term2 = 100
Kp_term3 = 100
Kp_term4 = 100

Kv_term1 = 20
Kv_term2 = 20
Kv_term3 = 20
Kv_term4 = 20

Kp = np.array([[Kp_term1, 0, 0, 0],
    [0, Kp_term2, 0, 0],
    [0, 0, Kp_term3, 0],
    [0, 0, 0, Kp_term4]])

Kv = np.array([[Kv_term1, 0, 0, 0],
    [0, Kv_term2, 0, 0],
    [0, 0, Kv_term3, 0],
    [0, 0, 0, Kv_term4]])


'''
Kp_term = 1600
Kv_term = 120
Kp = Kp_term*np.eye(4)
Kv = Kv_term*np.eye(4)
'''

# 9. Set Breakeven Torque ############### add multiplier for r in rxF
gm = 9.81
Tau1_max = 0.250*gm
Tau2_max = 0.134*gm
Tau3_max = 0.127*gm
Tau4_max = 0.087*gm

l1 = 0.180
d1 = l1
l2 = 0.300
l3 = 0.305
l4 = 0.071

m1 = 0.250
m2 = 0.134
m3 = 0.127
m4 = 0.087

'''
d1 = 0.4516
q2 = 0.7898
q3 = 0.0000
q4 = 1.5829
q5 = 0.3091
'''

'''
d1 = 0.000
q2 = 0.000
q3 = 0.000
q4 = 0.000
q5 = 0.000
'''

# 11. Set number of Joints
N = 4

# ##################### Trajectory Generator #####################
t1 = 0.2*tfinal # **0.2 = 2s
t2 = 0.8*tfinal # **0.8 = 8s

tc = (t1/tfinal)*(Iterations*TimeIncrement) #2s
tm = (t2/tfinal)*(Iterations*TimeIncrement) #8s
tf = 1*(Iterations*TimeIncrement) # **1 = 10s

# 12. Set setpoints here ########################################

#   Set Elevation and Angles ~~~~~~

#Fixed/Assigned
'''
q1des_main = 0.8#0.451640690281701348
q2des_main = -1.57#0.7898055239243771
q3des_main = -0.7854#-1.15
q4des_main = 0.7854#1.5828542923888724
q5des_main = -0.7854#0.30912280216854615
'''

'''
1
q1des_main = 0.8#0.451640690281701348
q2des_main = 0.7898#0.7898055239243771
q3des_main = 0.01#-1.15
q4des_main = 1.581#1.5828542923888724
q5des_main = 0.3091#0.30912280216854615
'''

'''
q1des_main = 0.4#0.451640690281701348
q2des_main = -6.52919#0.7898055239243771
q3des_main = -1.875985#-1.15
q4des_main = -0.00489955#1.5828542923888724
q5des_main = 0.00025769#0.30912280216854615
'''

'''
q1des_main = 1#0.451640690281701348
q2des_main = 3.14#0.7898055239243771
q3des_main = 3.14#-1.15
q4des_main = 0.34#1.5828542923888724
q5des_main = 1.57#0.30912280216854615
'''

'''
q1des_main = 0.4#0.451640690281701348
q2des_main = -3.14#0.7898055239243771
q3des_main = 0.00#-1.15
q4des_main = -2.84#1.5828542923888724
q5des_main = -2.18#0.30912280216854615
'''

q1des_main = -3.14#0.7898055239243771
q2des_main = 1.0472#-1.15
q3des_main = -2.84#1.5828542923888724
q4des_main = -2.18#0.30912280216854615

#to be populated
q1des = np.zeros((1,Iterations))
q2des = np.zeros((1,Iterations))
q3des = np.zeros((1,Iterations))
q4des = np.zeros((1,Iterations))
q1desPrev = 0
q2desPrev = 0
q3desPrev = 0
q4desPrev = 0

#   Set Velocities ~~~~~~
#Fixed
q1dotdes_max = 2*q1des_main/(2*tf-tc-(tf-tm))
q2dotdes_max = 2*q2des_main/(2*tf-tc-(tf-tm))
q3dotdes_max = 2*q3des_main/(2*tf-tc-(tf-tm))
q4dotdes_max = 2*q4des_main/(2*tf-tc-(tf-tm))

#to be populated
q1dotdes = np.zeros((1,Iterations))
q2dotdes = np.zeros((1,Iterations))
q3dotdes = np.zeros((1,Iterations))
q4dotdes = np.zeros((1,Iterations))

q1dotdesPrev = 0
q2dotdesPrev = 0
q3dotdesPrev = 0
q4dotdesPrev = 0

#   Set Accelerations ~~~~~~
#Fixed
q1dotdotdes_max = q1dotdes_max/tc
q2dotdotdes_max = q2dotdes_max/tc
q3dotdotdes_max = q3dotdes_max/tc
q4dotdotdes_max = q4dotdes_max/tc

#to be populated
q1dotdotdes = np.zeros((1,Iterations))
q2dotdotdes = np.zeros((1,Iterations))
q3dotdotdes = np.zeros((1,Iterations))
q4dotdotdes = np.zeros((1,Iterations))

#   Intersection ~~~~~~
qc1 = 0.5*q1dotdotdes_max*tc**2
qm1 = q1des_main - 0.5*q1dotdotdes_max*(tf-tm)**2

qc2 = 0.5*q2dotdotdes_max*tc**2
qm2 = q2des_main - 0.5*q2dotdotdes_max*(tf-tm)**2

qc3 = 0.5*q3dotdotdes_max*tc**2
qm3 = q3des_main - 0.5*q3dotdotdes_max*(tf-tm)**2

qc4 = 0.5*q4dotdotdes_max*tc**2
qm4 = q4des_main - 0.5*q4dotdotdes_max*(tf-tm)**2

#   Generate Setpoints ~~~~~~
p_index = np.arange(0, Iterations)
for p in p_index:
    tcurr = (p)*TimeIncrement
    print('tcurr')
    print(tcurr)
    if tcurr < tc: #2s
        # Acceleration
        q1dotdotdes[0,p] = q1dotdotdes_max
        q2dotdotdes[0,p] = q2dotdotdes_max
        q3dotdotdes[0,p] = q3dotdotdes_max
        q4dotdotdes[0,p] = q4dotdotdes_max

        # Velocity
        q1dotdes[0,p] = q1dotdotdes_max*tcurr
        q2dotdes[0,p] = q2dotdotdes_max*tcurr
        q3dotdes[0,p] = q3dotdotdes_max*tcurr
        q4dotdes[0,p] = q4dotdotdes_max*tcurr

        # Elevation/Angle
        q1des[0,p] = 0.5*q1dotdotdes_max*tcurr**2
        q2des[0,p] = 0.5*q2dotdotdes_max*tcurr**2
        q3des[0,p] = 0.5*q3dotdotdes_max*tcurr**2
        q4des[0,p] = 0.5*q4dotdotdes_max*tcurr**2
    elif (tcurr >= tc) and (tcurr <= tm): #between #2s and #8s
        # Acceleration
        q1dotdotdes[0,p] = 0
        q2dotdotdes[0,p] = 0
        q3dotdotdes[0,p] = 0
        q4dotdotdes[0,p] = 0

        # Velocity
        q1dotdes[0,p] = q1dotdes_max
        q2dotdes[0,p] = q2dotdes_max
        q3dotdes[0,p] = q3dotdes_max
        q4dotdes[0,p] = q4dotdes_max

        # Elevation/Angle
        q1des[0,p] = (qm1-qc1)/(tm-tc)*(tcurr) - qc1
        q2des[0,p] = (qm2-qc2)/(tm-tc)*(tcurr) - qc2
        q3des[0,p] = (qm3-qc3)/(tm-tc)*(tcurr) - qc3
        q4des[0,p] = (qm4-qc4)/(tm-tc)*(tcurr) - qc4
    else:
        q1dotdotdes[0,p] = -q1dotdotdes_max
        q2dotdotdes[0,p] = -q2dotdotdes_max
        q3dotdotdes[0,p] = -q3dotdotdes_max
        q4dotdotdes[0,p] = -q4dotdotdes_max

        # Velocity
        q1dotdes[0,p] = q1dotdotdes_max*(tf-tcurr)
        q2dotdes[0,p] = q2dotdotdes_max*(tf-tcurr)
        q3dotdes[0,p] = q3dotdotdes_max*(tf-tcurr)
        q4dotdes[0,p] = q4dotdotdes_max*(tf-tcurr)

        # Elevation/Angle
        q1des[0,p] = q1des_main - 0.5*q1dotdotdes_max*(tf-tcurr)**2
        q2des[0,p] = q2des_main - 0.5*q2dotdotdes_max*(tf-tcurr)**2
        q3des[0,p] = q3des_main - 0.5*q3dotdotdes_max*(tf-tcurr)**2
        q4des[0,p] = q4des_main - 0.5*q4dotdotdes_max*(tf-tcurr)**2


'''
print('q1des') #(5,1)
print(q1des.shape)
print('q1dotdes') #(5,5)
print(q1dotdes.shape)
print('q1dotdotdes') #(5,25)
print(q1dotdotdes.shape)

print('q1des.T') #(5,1)
print(np.transpose(q1des).shape)


#plt.plot((np.arange(0, Iterations)[np.newaxis]),np.array(q1des))
#plt.plot(np.array(q1des))
plt.plot(np.transpose(q5dotdotdes))
plt.show()
'''


# 13. Initialize Mass/Inertial, Coriolis/Centripetal, Gravity Matrices
Dnk = np.zeros((4,4))
Dnkj = np.zeros((4,4,4))
Gn = np.transpose(np.zeros(4)[np.newaxis])
Dnk_des = np.zeros((4,4))
Dnkj_des = np.zeros((4,4,4))
Gn_des = np.transpose(np.zeros(4)[np.newaxis])

# 14. Set Initial Condition same sa x0 at t = 0
y0 = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])


o_index = np.arange(0, Iterations)

for o in o_index: # Set Iterations
    print("o #############################################")
    print(o)
    # 15. Set Global Variables
    global Tau
    #global DA
    #global DB
    #global G

    # 16. Setpoint handover from Trajectory Generation
    q1_iter = y0[0]
    q2_iter = y0[1]
    q3_iter = y0[2]
    q4_iter = y0[3]

    # 17. Compute Error
    err1 = q1des[0,o] - y0[0]
    err2 = q2des[0,o] - y0[1]
    err3 = q3des[0,o] - y0[2]
    err4 = q4des[0,o] - y0[3]

    err1dot = q1dotdes[0,o] - y0[4]
    err2dot = q2dotdes[0,o] - y0[5]
    err3dot = q3dotdes[0,o] - y0[6]
    err4dot = q4dotdes[0,o] - y0[7]

    # 18. Compute Matrices (d1, q1, q2, q3, q4, m1, m2, m3, m4, l2, l3, l4)
    Dnk = DNK(d1, q1_iter, q2_iter, q3_iter, q4_iter, m1, m2, m3, m4, l2, l3, l4)
    Dnkj = DNKJ(d1, q1_iter, q2_iter, q3_iter, q4_iter, m1, m2, m3, m4, l2, l3, l4)
    Gn = GN(d1, q1_iter, q2_iter, q3_iter, q4_iter, m1, m2, m3, m4, l2, l3, l4)
    
    print('main')
    print('Gn') #(5,1)
    print(Gn.shape)
    print('Dnk') #(5,5)
    print(Dnk.shape)
    print('Dnkj') #(5,5,5)
    print(Dnkj.shape)
    

    Dnk_des = DNK(d1,q1des[0,o], q2des[0,o], q3des[0,o], q4des[0,o], m1, m2, m3, m4, l2, l3, l4)
    Dnkj_des = DNKJ(d1,q1des[0,o], q2des[0,o], q3des[0,o], q4des[0,o], m1, m2, m3, m4, l2, l3, l4)
    Gn_des = GN(d1,q1des[0,o], q2des[0,o], q3des[0,o], q4des[0,o], m1, m2, m3, m4, l2, l3, l4)

    
    print('Gn_des') #(5,1)
    print(Gn_des.shape)
    print('Dnk_des') #(5,5)
    print(Dnk_des.shape)
    print('Dnkj_des') #(5,5,5)
    print(Dnkj_des.shape)
    

    # 19. Hand-over Variables

    DA = np.array([[Dnk[0,0], Dnk[0,1], Dnk[0,2], Dnk[0,3]],
       [Dnk[1,0], Dnk[1,1], Dnk[1,2], Dnk[1,3]],
       [Dnk[2,0], Dnk[2,1], Dnk[2,2], Dnk[2,3]],
       [Dnk[3,0], Dnk[3,1], Dnk[3,2], Dnk[3,3]]])

    DB = np.array([[Dnkj[0,0,0], Dnkj[0,0,1], Dnkj[0,0,2], Dnkj[0,0,3], Dnkj[0,1,0], Dnkj[0,1,1], Dnkj[0,1,2], Dnkj[0,1,3], Dnkj[0,2,0], Dnkj[0,2,1], Dnkj[0,2,2], Dnkj[0,2,3], Dnkj[0,3,0], Dnkj[0,3,1], Dnkj[0,3,2], Dnkj[0,3,3]],
      [Dnkj[1,0,0], Dnkj[1,0,1], Dnkj[1,0,2], Dnkj[1,0,3], Dnkj[1,1,0], Dnkj[1,1,1], Dnkj[1,1,2], Dnkj[1,1,3], Dnkj[1,2,0], Dnkj[1,2,1], Dnkj[1,2,2], Dnkj[1,2,3], Dnkj[1,3,0], Dnkj[1,3,1], Dnkj[1,3,2], Dnkj[1,3,3]],
      [Dnkj[2,0,0], Dnkj[2,0,1], Dnkj[2,0,2], Dnkj[2,0,3], Dnkj[2,1,0], Dnkj[2,1,1], Dnkj[2,1,2], Dnkj[2,1,3], Dnkj[2,2,0], Dnkj[2,2,1], Dnkj[2,2,2], Dnkj[2,2,3], Dnkj[2,3,0], Dnkj[2,3,1], Dnkj[2,3,2], Dnkj[2,3,3]],
      [Dnkj[3,0,0], Dnkj[3,0,1], Dnkj[3,0,2], Dnkj[3,0,3], Dnkj[3,1,0], Dnkj[3,1,1], Dnkj[3,1,2], Dnkj[3,1,3], Dnkj[3,2,0], Dnkj[3,2,1], Dnkj[3,2,2], Dnkj[3,2,3], Dnkj[3,3,0], Dnkj[3,3,1], Dnkj[3,3,2], Dnkj[3,3,3]]])

    G = np.array(Gn)


    DA_des = np.array([[Dnk_des[0,0], Dnk_des[0,1], Dnk_des[0,2], Dnk_des[0,3]],
       [Dnk_des[1,0], Dnk_des[1,1], Dnk_des[1,2], Dnk_des[1,3]],
       [Dnk_des[2,0], Dnk_des[2,1], Dnk_des[2,2], Dnk_des[2,3]],
       [Dnk_des[3,0], Dnk_des[3,1], Dnk_des[3,2], Dnk_des[3,3]]])

    DB_des = np.array([[Dnkj_des[0,0,0], Dnkj_des[0,0,1], Dnkj_des[0,0,2], Dnkj_des[0,0,3], Dnkj_des[0,1,0], Dnkj_des[0,1,1], Dnkj_des[0,1,2], Dnkj_des[0,1,3], Dnkj_des[0,2,0], Dnkj_des[0,2,1], Dnkj_des[0,2,2], Dnkj_des[0,2,3], Dnkj_des[0,3,0], Dnkj_des[0,3,1], Dnkj_des[0,3,2], Dnkj_des[0,3,3]],
      [Dnkj_des[1,0,0], Dnkj_des[1,0,1], Dnkj_des[1,0,2], Dnkj_des[1,0,3], Dnkj_des[1,1,0], Dnkj_des[1,1,1], Dnkj_des[1,1,2], Dnkj_des[1,1,3], Dnkj_des[1,2,0], Dnkj_des[1,2,1], Dnkj_des[1,2,2], Dnkj_des[1,2,3], Dnkj_des[1,3,0], Dnkj_des[1,3,1], Dnkj_des[1,3,2], Dnkj_des[1,3,3]],
      [Dnkj_des[2,0,0], Dnkj_des[2,0,1], Dnkj_des[2,0,2], Dnkj_des[2,0,3], Dnkj_des[2,1,0], Dnkj_des[2,1,1], Dnkj_des[2,1,2], Dnkj_des[2,1,3], Dnkj_des[2,2,0], Dnkj_des[2,2,1], Dnkj_des[2,2,2], Dnkj_des[2,2,3], Dnkj_des[2,3,0], Dnkj_des[2,3,1], Dnkj_des[2,3,2], Dnkj_des[2,3,3]],
      [Dnkj_des[3,0,0], Dnkj_des[3,0,1], Dnkj_des[3,0,2], Dnkj_des[3,0,3], Dnkj_des[3,1,0], Dnkj_des[3,1,1], Dnkj_des[3,1,2], Dnkj_des[3,1,3], Dnkj_des[3,2,0], Dnkj_des[3,2,1], Dnkj_des[3,2,2], Dnkj_des[3,2,3], Dnkj_des[3,3,0], Dnkj_des[3,3,1], Dnkj_des[3,3,2], Dnkj_des[3,3,3]]])

    G_des = np.array(Gn_des)



    qdotdotDes = np.array([[q1dotdotdes[0,o]],
                 [q2dotdotdes[0,o]],
                 [q3dotdotdes[0,o]],
                 [q4dotdotdes[0,o]]])

    qdotDes = np.array([[q1dotdes[0,o]*q1dotdes[0,o]],
               [q1dotdes[0,o]*q2dotdes[0,o]],
               [q1dotdes[0,o]*q3dotdes[0,o]],
               [q1dotdes[0,o]*q4dotdes[0,o]],
               [q2dotdes[0,o]*q1dotdes[0,o]],
               [q2dotdes[0,o]*q2dotdes[0,o]],
               [q2dotdes[0,o]*q3dotdes[0,o]],
               [q2dotdes[0,o]*q4dotdes[0,o]],
               [q3dotdes[0,o]*q1dotdes[0,o]],
               [q3dotdes[0,o]*q2dotdes[0,o]],
               [q3dotdes[0,o]*q3dotdes[0,o]],
               [q3dotdes[0,o]*q4dotdes[0,o]],
               [q4dotdes[0,o]*q1dotdes[0,o]],
               [q4dotdes[0,o]*q2dotdes[0,o]],
               [q4dotdes[0,o]*q3dotdes[0,o]],
               [q4dotdes[0,o]*q4dotdes[0,o]]])

    
    print('qdotdotDes') #(5,1)
    print(qdotdotDes.shape)
    print('qdotDes') #(25,1)
    print(qdotDes.shape)
    

    ############# Compute Error and Output Torque ##############
    err = np.transpose(np.array([err1, err2, err3, err4])[np.newaxis]);
    errdot = np.transpose(np.array([err1dot, err2dot, err3dot, err4dot])[np.newaxis]);

    Tau = np.zeros((4,1));

    
    print('G_des') #(5,1)
    print(G_des.shape)
    print('DA_des') #(5,5)
    print(DA_des.shape)
    print('DB_des') #(5,25)
    print(DB_des.shape)
    print('np.matmul(Kv,errdot)') #(5, 1)
    print((np.matmul(Kv,errdot)).shape)
    print('np.matmul(Kp,err)') #(5, 1)
    print((np.matmul(Kp,err)).shape)

    Tau = (G_des + np.matmul(DA_des,qdotdotDes) + np.matmul(DB_des,qdotDes)) + np.matmul(DA,(np.matmul(Kv,errdot)+np.matmul(Kp,err)))


    print('Tau') #(5, 1)
    print(Tau.shape)

    ###############################################################
    ##### Add torque limiting here ################~~~~~~~~~~~~~~~~
    ###############################################################

    '''
    if Tau(1,1) >= Tau1_max
        Tau(1,1) = Tau1_max;
    elseif Tau(1,1) < 0
        Tau(1,1) = 0;
    else
        Tau(1,1) = Tau(1,1);
    end

    if Tau(2,1) >= Tau2_max
        Tau(2,1) = Tau2_max;
    elseif Tau(2,1) < 0
        Tau(2,1) = 0;
    else
        Tau(2,1) = Tau(2,1);
    end

    if Tau(3,1) >= Tau3_max
        Tau(3,1) = Tau3_max;
    elseif Tau(3,1) < 0
        Tau(3,1) = 0;
    else
        Tau(3,1) = Tau(3,1);
    end

    if Tau(4,1) >= Tau4_max
        Tau(4,1) = Tau4_max;
    elseif Tau(4,1) < 0
        Tau(4,1) = 0;
    else
        Tau(4,1) = Tau(4,1);
    end

    if Tau(5,1) >= Tau5_max
        Tau(5,1) = Tau5_max;
    elseif Tau(5,1) < 0
        Tau(5,1) = 0;
    else
        Tau(5,1) = Tau(5,1);
    end

    Tau_clipped = Tau
    '''

    ####################### Simulation Proper #######################

    # 20. Set the initial conditions input to lsode (if 1st start, initialize to start position then get next values for looping)
    if o==1:
        x0 = np.transpose([0.0, 0.0, 0.0, 0.0, 0.00, 0.00, 0.00, 0.00]) #Nest
    else:
        x0 = np.transpose([y0[0], y0[1], y0[2], y0[3], y0[4], y0[5], y0[6], y0[7]])
    

    t_plot = np.linspace(0, tfinal, Iterations)
    y0_out = odeint(xdotTeleNoPID, x0, np.array([0, TimeIncrement])) #args = (,)
    #[t, y0_out] = ode45(@xdotTeleNoPID,[0 TimeIncrement],x0); #Matlab
    #y0_out = lsode("xdotTeleNoPID", x0, [0, TimeIncrement]); #Octave, TimeIncrement = 10^-2


    # 21. Store Values for Plotting
    y0 = y0_out[-1,:] #get last value
    y0_store[o,:] = y0 #store for plotting

    Tau1_store[o,:] = Tau[0,0]
    Tau2_store[o,:] = Tau[1,0]
    Tau3_store[o,:] = Tau[2,0]
    Tau4_store[o,:] = Tau[3,0]

    err1_store[o,:] = err1
    err2_store[o,:] = err2
    err3_store[o,:] = err3
    err4_store[o,:] = err4

    q1_store[o,:] = q1_iter
    q2_store[o,:] = q2_iter
    q3_store[o,:] = q3_iter
    q4_store[o,:] = q4_iter

# create 2 subplots
fig, ax = plt.subplots(nrows=1, ncols=4)
x = np.linspace(0, tfinal, Iterations)

print('q1_store') #(5,1)
print(q1_store.shape)

print('x') #(5,1)
print(x.shape)

print('q1des') #(5,1)
print(q1des.shape)

ax[0].plot(x, q1_store, label = "q")
q1des_mat = q1des.reshape((100, 1))
ax[0].plot(x, q1des_mat, label = "q_des")
ax[1].plot(x, q2_store, label = "q")
q2des_mat = q2des.reshape((100, 1))
ax[1].plot(x, q2des_mat, label = "q_des")
ax[2].plot(x, q3_store, label = "q")
q3des_mat = q3des.reshape((100, 1))
ax[2].plot(x, q3des_mat, label = "q_des")
ax[3].plot(x, q4_store, label = "q")
q4des_mat = q4des.reshape((100, 1))
ax[3].plot(x, q4des_mat, label = "q_des")

# plot 2 subplots
ax[0].set_title('q1')
ax[1].set_title('q2')
ax[2].set_title('q3')
ax[3].set_title('q4')
 
fig.suptitle('Joint movement of robot arm')
plt.show()

######################## Animate Arm ############################
# to run GUI event loop
plt.ion()
fig = plt.figure() #figsize = (8,8) #https://matplotlib.org/2.1.1/api/_as_gen/matplotlib.pyplot.plot.html

ax = plt.axes(projection='3d')
ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.legend(['l1','l2','l3','l4'])

x = np.array([0.0, 0.0])
y = np.array([0.0, 0.0])
z = np.array([0.0, 0.0])

line1, = ax.plot(x, y, z)
line2, = ax.plot(x, y, z)
line3, = ax.plot(x, y, z)
line4, = ax.plot(x, y, z)

i_index = np.arange(0, Iterations)#np.arange(1, len(y0_store))
for i in i_index:

    q1 = y0_store[i,0]
    q2 = y0_store[i,1]
    q3 = y0_store[i,2]
    q4 = y0_store[i,3]

    A01 = [[np.cos(q1), 0, np.sin(q1), 0],
           [np.sin(q1), 0, -np.cos(q1), 0],
           [0, 1, 0, d1],
           [0, 0, 0, 1]]
    A12 = [[np.cos(q2), -np.sin(q2), 0, l2*np.cos(q2)],
           [np.sin(q2), np.cos(q2), 0, l2*np.sin(q2)],
           [0, 0, 1, 0],
           [0, 0, 0, 1]]
    A23 = [[np.cos(q3), -np.sin(q3), 0, l3*np.cos(q3)],
           [np.sin(q3), np.cos(q3), 0, l3*np.sin(q3)],
           [0, 0, 1, 0],
           [0, 0, 0, 1]]
    A34 = [[np.cos(q4), -np.sin(q4), 0, l4*np.cos(q4)],
           [np.sin(q4), np.cos(q4), 0, l4*np.sin(q4)],
           [0, 0, 1, 0],
           [0, 0, 0, 1]]

    #Get end locations per joint
    L1_end = np.matmul(A01,np.transpose(np.array([0, 0, 0, 1])[np.newaxis]))
    L2_end = np.matmul(np.matmul(A01,A12),np.transpose(np.array([0, 0, 0, 1])[np.newaxis]))
    L3_end = np.matmul(np.matmul(np.matmul(A01,A12),A23),np.transpose(np.array([0, 0, 0, 1])[np.newaxis]))
    L4_end = np.matmul(np.matmul(np.matmul(np.matmul(A01,A12),A23),A34),np.transpose(np.array([0, 0, 0, 1])[np.newaxis]))



    x1 = L1_end[0].item()
    y1 = L1_end[1].item()
    z1 = L1_end[2].item()

    x2 = L2_end[0].item()
    y2 = L2_end[1].item()
    z2 = L2_end[2].item()

    x3 = L3_end[0].item()
    y3 = L3_end[1].item()
    z3 = L3_end[2].item()

    x4 = L4_end[0].item()
    y4 = L4_end[1].item()
    z4 = L4_end[2].item()

    
    q1plotX = np.array([0.0, x1])
    q1plotY = np.array([0.0, y1])
    q1plotZ = np.array([0.0, z1])

    q2plotX = np.array([x1, x2])
    q2plotY = np.array([y1, y2])
    q2plotZ = np.array([z1, z2])

    q3plotX = np.array([x2, x3])
    q3plotY = np.array([y2, y3])
    q3plotZ = np.array([z2, z3])

    q4plotX = np.array([x3, x4])
    q4plotY = np.array([y3, y4])
    q4plotZ = np.array([z3, z4])


    '''
    ax.plot(q1plotX,q1plotY,q1plotZ, 'b-') #https://www.tutorialspoint.com/how-do-you-create-line-segments-between-two-points-in-matplotlib

    ax.plot(q2plotX,q2plotY,q2plotZ, 'k-')
    ax.plot(q3plotX,q3plotY,q3plotZ, 'r-')
    ax.plot(q4plotX,q4plotY,q4plotZ, 'g-')
    ax.plot(q5plotX,q5plotY,q5plotZ, 'm-')
    '''
    '''
    line1.set_xdata(q1plotX)
    line1.set_ydata(q1plotY)
    line1.set_zdata(q1plotZ)

    line2.set_xdata(q2plotX)
    line2.set_ydata(q2plotY)
    line2.set_zdata(q2plotZ)

    line3.set_xdata(q3plotX)
    line3.set_ydata(q3plotY)
    line3.set_zdata(q3plotZ)

    line4.set_xdata(q4plotX)
    line4.set_ydata(q4plotY)
    line4.set_zdata(q4plotZ)

    line5.set_xdata(q5plotX)
    line5.set_ydata(q5plotY)
    line5.set_zdata(q5plotZ)
    '''
    line1.set_data_3d(q1plotX, q1plotY, q1plotZ)

    line2.set_data_3d(q2plotX, q2plotY, q2plotZ)

    line3.set_data_3d(q3plotX, q3plotY, q3plotZ)

    line4.set_data_3d(q4plotX, q4plotY, q4plotZ)

    fig.canvas.draw() #https://stackoverflow.com/questions/30972289/draw-now-and-matplotlib
    fig.canvas.flush_events()
    time.sleep(TimeIncrement)
    