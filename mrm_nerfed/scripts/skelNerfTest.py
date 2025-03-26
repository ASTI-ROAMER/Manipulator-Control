#!/usr/bin/env python

from std_msgs.msg import Float64, Float32MultiArray, MultiArrayLayout, MultiArrayDimension

import sys
import os
import threading
import math
import time
import faulthandler
faulthandler.enable()

import numpy as np
import ctypes
import struct
import cv2
import rospy
import matplotlib.pyplot as plt 
from cv_bridge import CvBridge, CvBridgeError

from nav_msgs.msg import Odometry as msg_Odom
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import CameraInfo as msg_CamInfo
from std_msgs.msg import Header
import message_filters
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

import ros_numpy
import random
import matplotlib.pyplot as plt
from scipy import spatial

import tf2_ros
import tf2_py as tf2
from sensor_msgs.msg import PointCloud2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from tf2_geometry_msgs import do_transform_pose
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point, Pose
import tf2_geometry_msgs

import open3d as o3d
from open3d_ros_helper import open3d_ros_helper as orh
from sklearn.cluster import KMeans, DBSCAN, OPTICS, AffinityPropagation, MeanShift, estimate_bandwidth, spectral_clustering, AgglomerativeClustering, Birch, BisectingKMeans
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KDTree
import scipy.spatial.distance

from tqdm import tqdm

import numpy as np
import plotly.graph_objects as go

import open3d as o3d
import numpy as np
from pc_skeletor import LBC
from pc_skeletor import SLBC
from open3d_ros_helper import open3d_ros_helper as orh



############################################# Perception: OPENCV/PCL STUFF ##########################################

nearestPoints_global = np.array([])

def lidarCallback(ros_data):
    global nearestPoints_global
    try:
        # Start: Get PCL data

        #ROS to PCL ----------------
        scanHeader = ros_data.header.stamp
        scanFrameID = ros_data.header.frame_id
        scanWidth = ros_data.width
        scanRowStep = ros_data.row_step

        print("scanWidth")
        print(scanWidth)
        print("scanFrameID")
        print(scanFrameID)

        # 1. Do REPOSITIONING OF POINT CLOUDS using tf2 from camera frame to base link
        scanCloud_ROS_map2base = PointCloud2()

        try:
            transform = tfBuffer.lookup_transform('base_link', scanFrameID, rospy.Time(0)) #, rospy.Duration(timeout) "realsense_depth_optical_frame"
            #transform = tfBuffer.lookup_transform('base_link', 'realsense_depth_optical_frame', scanHeader) #, rospy.Duration(timeout) "realsense_depth_optical_frame"
        except tf2.LookupException as ex:
            rospy.logwarn(ex)
            print("No new messages 1")
            return
        except tf2.ExtrapolationException as ex:
            rospy.logwarn(ex)
            print("No new messages 2")
            return


        scanCloud_ROS_map2base = do_transform_cloud(ros_data, transform)


        #   convert ROS message to Open3D ---------------------- https://github.com/SeungBack/open3d-ros-helper
        scanCloud_o3d = orh.rospc_to_o3dpc(scanCloud_ROS_map2base, remove_nans=True) #For realsense depth input

        #   downsample pointcloud with a voxel size of 0.01 meters
        scanCloud_o3d_voxel = scanCloud_o3d.voxel_down_sample(voxel_size=0.01) #


        # 2. Do POINT CLOUD SLICING ######################
        
        #   crop along Y-axis
        bounding_box  = scanCloud_o3d_voxel.get_axis_aligned_bounding_box()
        bounding_box_points = np.asarray(bounding_box.get_box_points())
        bounding_box_points[:,1] = np.clip(bounding_box_points[:,1], a_min = 0, a_max = 1.2)
        bounding_box_cropped =  o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bounding_box_points))
        scanCloud_o3d_cropped = scanCloud_o3d_voxel.crop(bounding_box_cropped)

        #   crop along X_axis
        bounding_box  = scanCloud_o3d_cropped.get_axis_aligned_bounding_box()
        bounding_box_points = np.asarray(bounding_box.get_box_points())
        bounding_box_points[:,0] = np.clip(bounding_box_points[:,0], a_min = -1.5, a_max = 1.5)
        bounding_box_cropped =  o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bounding_box_points))
        scanCloud_o3d_cropped = scanCloud_o3d_voxel.crop(bounding_box_cropped)

        #   ground removal
        plane_model, inliers = scanCloud_o3d_cropped.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        inlier_cloud = scanCloud_o3d_cropped.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = scanCloud_o3d_cropped.select_by_index(inliers, invert=True)
        scanCloud_o3d_cropped = outlier_cloud

        #mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        #o3d.visualization.draw_geometries([mesh_frame, inlier_cloud, outlier_cloud])

        # 3. Do COMPUTE NORMALS
        scanCloud_o3d_cropped.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)) #The two key arguments radius = 0.1 and max_nn = 30 specifies search radius and maximum nearest neighbor. It has 10cm of search radius, and only considers up to 30 neighbors to save computation time.
        scanCloud_o3d_cropped.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.3]))

        # 4. Do SOR (Statistical Outlier Filter)
        nearest_neighbor = 7 #16
        std_multiplier = 2 # 10 Use standard dev of point dist and apply a multiplier. If above outlier if beneath inlier 

        filtered_pcd = scanCloud_o3d_cropped.remove_statistical_outlier(nearest_neighbor,std_multiplier)

        outliers = scanCloud_o3d_cropped.select_by_index(filtered_pcd[1], invert = True) #[1] to return indexes of outliers and not point clouds
        outliers.paint_uniform_color([0,0,1])
        scanCloud_o3d_cropped = filtered_pcd[0] #Get pointcloud with only inliers
 
        pcd_center = scanCloud_o3d_cropped.get_center()
        scanCloud_center = o3d.geometry.PointCloud()
        scanCloud_center.points = o3d.utility.Vector3dVector(np.asarray([pcd_center]))


        # Open3D -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

        # 5. Ball Pivoting - Mesh generation
        #       a. Set the radius of the ball to the average distance between points in the cloud
        dist_nnd = scanCloud_o3d_cropped.compute_nearest_neighbor_distance()
        mean_dist = np.mean(dist_nnd)
        radius = 3*(mean_dist)
        #       b. Ball pivoting
        rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            scanCloud_o3d_cropped, o3d.utility.DoubleVector([radius, radius*5]))

        #       c. Orient triangles
        if rec_mesh.is_orientable() == True:
            print("rec_mesh.is_orientable()")
            print(rec_mesh.is_orientable())
            print("rec_mesh.orient_triangles()")
            print(rec_mesh.orient_triangles())
        else:
            print("rec_mesh.is_orientable()")
            print(rec_mesh.is_orientable())
            print("rec_mesh.orient_triangles()")
            print(rec_mesh.orient_triangles())

        #       d. Decimate the mesh to minimize triangles  
        rec_mesh = rec_mesh.simplify_quadric_decimation(10000)
        rec_mesh.compute_vertex_normals()

        #       e. Clean up mesh
        rec_mesh = rec_mesh.filter_smooth_simple(number_of_iterations = 1)
        rec_mesh.compute_vertex_normals()
        rec_mesh = rec_mesh.remove_duplicated_vertices()
        rec_mesh.compute_vertex_normals()
        rec_mesh = rec_mesh.remove_unreferenced_vertices()
        rec_mesh.compute_vertex_normals()
        rec_mesh = rec_mesh.remove_degenerate_triangles()
        rec_mesh.compute_vertex_normals()
        rec_mesh = rec_mesh.remove_duplicated_triangles()
        rec_mesh.compute_vertex_normals()
        rec_mesh = rec_mesh.remove_non_manifold_edges()
        rec_mesh.compute_vertex_normals()

        print("rec_mesh.get_non_manifold_vertices()")
        print(rec_mesh.get_non_manifold_vertices())
        print("rec_mesh.get_non_manifold_edges(allow_boundary_edges=True)")
        print(rec_mesh.get_non_manifold_edges(allow_boundary_edges=True))
        print("rec_mesh.is_vertex_manifold()")
        print(rec_mesh.is_vertex_manifold())
        print("rec_mesh.is_edge_manifold(allow_boundary_edges=True)")
        print(rec_mesh.is_edge_manifold(allow_boundary_edges=True))

        #       f. Convert vertices and faces to mesh via numpy
        verts = rec_mesh.vertices
        faces = rec_mesh.triangles
        scanCloud_o3d_mesh = rec_mesh.sample_points_uniformly(number_of_points=2000)

        # PC Skeletor -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
        
        # 6. Skeletonization of Pointcloud

        #       a. Laplacian Based Contraction
        
        pcd_sum = scanCloud_o3d_mesh + scanCloud_o3d_cropped

        lbc = LBC(point_cloud=pcd_sum,
                  down_sample=0.01)
        es = lbc.extract_skeleton() #o3d.geometry.PointCloud
        et = lbc.extract_topology() #o3d.geometry.LineSet
        ec = np.array([])

        #       b. Sort line segments according to line midpoint z location

        rng = np.random.default_rng()

        etP = np.asarray(et.points)
        etL = np.asarray(et.lines)

        for i in np.arange(0,len(et.lines)):
            if i == 0:
                ec = rng.random((1,3))
            else:
                ec = np.vstack((ec, rng.random((1,3))))


        #       c. Get length per line
        lineLen = np.array([])
        for i in np.arange(0,len(et.lines)):

            ptA = etL[i][0]
            ptB = etL[i][1]
            distTemp = np.sqrt( (etP[ptB][0]-etP[ptA][0])**2 + (etP[ptB][1]-etP[ptA][1])**2 + (etP[ptB][2]-etP[ptA][2])**2  )
            lineLen = np.append(lineLen, distTemp)

        #       d. Remove lines shorter than 1 foot https://stackoverflow.com/questions/30679192/how-to-remove-multiple-values-from-an-array-at-once
        idxRemove = np.ravel([np.where(lineLen < 0.05)]) #0.16 Banana, 0.05 potted plant

        #       e. Get midpoint location per line
        lineMidpoint = np.array([])
        for i in np.arange(0,len(et.lines)):

            ptA = etL[i][0]
            ptB = etL[i][1]
            midTemp = (etP[ptB][2]+etP[ptA][2])/2 # just get z-axis
            lineMidpoint = np.append(lineMidpoint, midTemp)

        lineMidpoint_filt = np.delete(lineMidpoint, idxRemove)

        print(np.argsort(lineMidpoint_filt)) #Sort from lowest leaf to highest
        print(np.flip(np.argsort(lineMidpoint_filt))) #Sort from highest leaf to lowest

        #       f. Filter points as well
        etP_filt = np.delete(etP, idxRemove, axis=0)
        etL_filt = np.delete(etL, idxRemove, axis=0)

        #       g. Split points into segments
        segments = 5
        lineSplitPoints = np.zeros((len(etL_filt), 3, 5))
        for i in np.arange(0,len(etL_filt)):

            ptA = etL_filt[i][0]
            ptB = etL_filt[i][1]
            
            xDelta = (etP[ptB][0]-etP[ptA][0])/(segments-1)

            yDelta = (etP[ptB][1]-etP[ptA][1])/(segments-1)

            zDelta = (etP[ptB][2]-etP[ptA][2])/(segments-1)


            for j in np.linspace(0, segments-1, segments):
                if j == 0:
                    lineSplitPoints[i][0][int(j)] = etP[ptA][0]
                    lineSplitPoints[i][1][int(j)] = etP[ptA][1]
                    lineSplitPoints[i][2][int(j)] = etP[ptA][2]
                else:
                    lineSplitPoints[i][0][int(j)] = lineSplitPoints[i][0][int(j-1)] + xDelta
                    lineSplitPoints[i][1][int(j)] = lineSplitPoints[i][1][int(j-1)] + yDelta
                    lineSplitPoints[i][2][int(j)] = lineSplitPoints[i][2][int(j-1)] + zDelta

        #       h. Color segmented points
        lsp = lineSplitPoints.shape #lsp (29, 3, 5) nearestPoints (145, 3)

        lineSplitStacked = np.array([])
        for i in np.arange(0,lsp[0]):
            if i == 0:
                lineSplitStacked = lineSplitPoints[i][:][:]
            else:
                lineSplitStacked = np.append(lineSplitStacked, lineSplitPoints[i][:][:], axis = 1)

        lineSplitStacked = np.transpose(lineSplitStacked)

        lineSplitColor = np.array([])
        rng.random((1,3))
        for i in np.arange(0,lsp[0]):
            if i == 0:
                tempColor = rng.random((1,3)).tolist()[0]
                lineSplitColor = [tempColor,tempColor,tempColor,tempColor,tempColor]
            else:
                tempColor = rng.random((1,3)).tolist()[0]
                lineSplitColor = np.append(lineSplitColor, [tempColor,tempColor,tempColor,tempColor,tempColor], axis = 0)

        setPoints = o3d.geometry.PointCloud()
        setPoints.points = o3d.utility.Vector3dVector(lineSplitStacked)
        setPoints.colors = o3d.utility.Vector3dVector(lineSplitColor)

        skeleton = o3d.geometry.LineSet()
        skeleton.points = o3d.utility.Vector3dVector(et.points)
        skeleton.lines = o3d.utility.Vector2iVector(et.lines)
        skeleton.colors = o3d.utility.Vector3dVector(ec)


        #       i. Get closest point in cloud for each setpoint
        PX = lineSplitStacked[:,0]
        PY = lineSplitStacked[:,1]
        PZ = lineSplitStacked[:,2]

        cloudInput  =  np.asarray(scanCloud_o3d_cropped.points)

        CX = cloudInput[:,0]
        CY = cloudInput[:,1]
        CZ = cloudInput[:,2]

        nearestPoints = np.zeros((len(lineSplitColor),3))
        for i in np.arange(0,len(PX)):
            distances = np.sqrt( (CX-PX[i])**2 + (CY-PY[i])**2 + (CZ-PZ[i])**2  )
            idxMin = np.argmin(distances)
            if i == 0:
                nearestPoints[i,:] = np.asarray([CX[idxMin], CY[idxMin], CZ[idxMin]])

            else:
                nearestPoints[i,:] = np.asarray([CX[idxMin], CY[idxMin], CZ[idxMin]])


        setPoints_new = o3d.geometry.PointCloud()
        setPoints_new.points = o3d.utility.Vector3dVector(nearestPoints)
        setPoints_new.colors = o3d.utility.Vector3dVector(lineSplitColor)


        #       j. Draw line visuals for trajectories

        col1 = np.arange(0,len(nearestPoints))
        col2 = np.arange(1,len(nearestPoints)+1)

        col1 = np.matrix(col1).T
        col2 = np.matrix(col2).T

        nearestLines =  np.hstack((col1,col2))
        nearestLines = np.array(nearestLines)

        nearest = o3d.geometry.LineSet()
        nearest.points = o3d.utility.Vector3dVector(nearestPoints)
        nearest.colors = o3d.utility.Vector3dVector(lineSplitColor)
        nearest.lines = o3d.utility.Vector2iVector(nearestLines)

        nearestPoints_T = np.transpose(nearestPoints)

        nearestPoints_global = nearestPoints



        #       k. Get normal vector for each point
        setPoints_new.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)) #The two key arguments radius = 0.1 and max_nn = 30 specifies search radius and maximum nearest neighbor. It has 10cm of search radius, and only considers up to 30 neighbors to save computation time.
        setPoints_new.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.3]))


        print("Print a normal vector of the 0th point")
        print(setPoints_new.normals[0])
        print("Print the normal vectors of the first 10 points")
        print(np.asarray(setPoints_new.normals)[:10, :])
        print("")

        print("Print the first 10 points")
        print(np.asarray(setPoints_new.points)[:10, :])
        print("")

        #print(nearestPoints.shape) #lsp (29, 3, 5) nearestPoints (145, 3)


        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries([mesh_frame, pcd_sum, nearest, setPoints_new])

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries([mesh_frame, pcd_sum, setPoints_new])

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries([mesh_frame, setPoints_new])




        #o3d.visualization.draw_geometries([mesh_frame, nearest, setPoints_new, pcd_sum])


        #       l. Get the point at the end of each normal vector :: vectorAB = B-A, B = vectorAB + A

        pointA = np.asarray(setPoints_new.points)
        normalVector = np.asarray(setPoints_new.normals)

        pointB = np.add(normalVector,pointA)

        endPoints_new = o3d.geometry.PointCloud()
        endPoints_new.points = o3d.utility.Vector3dVector(pointB)
        endPoints_new.colors = o3d.utility.Vector3dVector(lineSplitColor)


        setPoints_pub = orh.o3dpc_to_rospc(setPoints_new, frame_id=scanCloud_ROS_map2base.header.frame_id, stamp=scanCloud_ROS_map2base.header.stamp)
        pub1.publish(setPoints_pub)
        endPoints_pub = orh.o3dpc_to_rospc(endPoints_new, frame_id=scanCloud_ROS_map2base.header.frame_id, stamp=scanCloud_ROS_map2base.header.stamp)
        pub2.publish(endPoints_pub)

        dists = endPoints_new.compute_point_cloud_distance(setPoints_new)
        dists = np.asarray(dists)

        print("distances")
        print(dists)

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries([mesh_frame, setPoints_new, endPoints_new])


        #       m. Get the end effector pair target points to align it to the normal vector 12.5cm/125mm from the leaf
        #       2|-----------------------|---End-effector-link-j->i-100mm---|--------Distance-from-leaf-125mm--------|1
        #       2|-----------------------j----------------1000mm/1meter-----i----------------------------------------|1

        point1 = pointA
        point2 = pointB

        dist_1i = 0.125
        dist_12 = 1.100

        dist_1j = 0.225

        ratio_i = dist_1i/dist_12
        ratio_j = dist_1j/dist_12

        point_i = np.add(point1,np.multiply(ratio_i,np.subtract(point2,point1)))
        point_j = np.add(point1,np.multiply(ratio_j,np.subtract(point2,point1)))

        print("pointi")
        print(point_i.shape)

        print("pointj")
        print(len(point_j))

        pointI = o3d.geometry.PointCloud()
        pointI.points = o3d.utility.Vector3dVector(point_i)
        pointI.colors = o3d.utility.Vector3dVector(lineSplitColor)
        pointI_pub = orh.o3dpc_to_rospc(pointI, frame_id=scanCloud_ROS_map2base.header.frame_id, stamp=scanCloud_ROS_map2base.header.stamp)
        pub3.publish(pointI_pub)

        pointJ = o3d.geometry.PointCloud()
        pointJ.points = o3d.utility.Vector3dVector(point_j)
        pointJ.colors = o3d.utility.Vector3dVector(lineSplitColor)
        pointJ_pub = orh.o3dpc_to_rospc(pointJ, frame_id=scanCloud_ROS_map2base.header.frame_id, stamp=scanCloud_ROS_map2base.header.stamp)
        pub4.publish(pointJ_pub)


    except CvBridgeError as e:
        print(e)
        return

############################################# Inverse Kinematics: SCIPY STUFF ##########################################



if __name__ == "__main__":

        ### C. Gazebo
    try: #https://answers.ros.org/question/244271/how-to-enter-the-callback-function-only-whenever-a-topic-is-updated-but-otherwise-keep-doing-something-else/
        rospy.init_node("depth_image_processor")
        topic1 = "/realsense/depth/color/points" #"/rtabmap/cloud_obstacles"
        topic1_out = "/pointA"  
        topic2_out = "/pointB"
        topic3_out = "/pointI"
        topic4_out = "/pointJ"

        #tfListener = tf.TransformListener()

        tfBuffer = tf2_ros.Buffer()
        tfListener = tf2_ros.TransformListener(tfBuffer)

        sub1 = rospy.Subscriber(topic1, PointCloud2, lidarCallback)
            
        #pub1 = rospy.Publisher(topic1_out, msg_Image, queue_size=20)
        pub1 = rospy.Publisher(topic1_out, PointCloud2, queue_size=20)
        pub2 = rospy.Publisher(topic2_out, PointCloud2, queue_size=20)
        pub3 = rospy.Publisher(topic3_out, PointCloud2, queue_size=20)
        pub4 = rospy.Publisher(topic4_out, PointCloud2, queue_size=20)
        pub5 = rospy.Publisher(topic4_out, Float32MultiArray, queue_size=20)


        rate = rospy.Rate(float(rospy.get_param('~rate', 30.0)))


        #with wait4Image:
        while not rospy.is_shutdown(): #https://answers.ros.org/question/310291/while-loop-only-executes-subscriber-callback/
            

            rate.sleep()

    except rospy.ROSInterruptException: #https://answers.ros.org/question/244271/how-to-enter-the-callback-function-only-whenever-a-topic-is-updated-but-otherwise-keep-doing-something-else/
        
        print("Nein")
        pass