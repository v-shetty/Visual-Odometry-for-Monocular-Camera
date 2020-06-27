#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
import math
import tf
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray



class Nodo(object):
    def __init__(self):
        # Params
         
        self.br = CvBridge()
        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(30)

        # Publishers
        self.img_pub = rospy.Publisher('/camera/image_raw', Image,queue_size=10)
        self.point_pub = rospy.Publisher('/myPoint' , Marker, queue_size=1)
        
        
        
        self.points = Marker()
        self.points.header.stamp = rospy.Time.now()
        self.points.header.frame_id = "/map"
        self.points.ns = "points_and_lines"
        self.points.action = Marker.ADD
        
        self.points.id = 0;
        self.points.type = Marker.LINE_STRIP;
        #LINE_STRIP POINTS
        
        
        self.points.scale.x = 3.1;
        self.points.scale.y = 3.1;
        #self.points.scale.z = 0.1;
        
        self.points.color.g = 1.0;
        self.points.color.a = 1.0;
        self.points.color.r = 0.0
        self.points.color.b = 0.0
        

        self.points.pose.orientation.w = 1.0
        
        
    def start(self):


        prev_image = None
        feature_detector = cv2.FastFeatureDetector_create(threshold=25,
                                                          nonmaxSuppression=True)

        lk_params = dict(winSize=(21, 21),
                         criteria=(cv2.TERM_CRITERIA_EPS |
                                   cv2.TERM_CRITERIA_COUNT, 30, 0.03))

        camera_matrix = np.array([[718.8560, 0.0, 607.1928],
                                  [0.0, 718.8560, 185.2157],
                                  [0.0, 0.0, 1.0]])




        current_pos = np.zeros((3, 1))
        current_rot = np.eye(3)
        
        br = CvBridge()

        index= 0


        path = r'/home/user/catkin_ws/src/publisher/src/point_demo/sequence_00'

        image_format_left = '{:06d}.png'
	
        
        

        
        while not rospy.is_shutdown():
         

            loc = os.path.join(path, image_format_left.format(index))
            print(loc)
	   
            image = cv2.imread(loc)
	   
            detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

            kps = detector.detect(image)

            kp = np.array([x.pt for x in kps], dtype=np.float32)

            if prev_image is None:
               prev_image = image
               prev_keypoint = kp
               continue

            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_image,
                                                   image, prev_keypoint,
                                                   None, **lk_params)

            E, mask = cv2.findEssentialMat(p1, prev_keypoint, camera_matrix,
                                           cv2.RANSAC, 0.999, 1.0, None)

            points, R, t, mask = cv2.recoverPose(E, p1, prev_keypoint, camera_matrix)
            
            scale = 1.0

            current_pos += current_rot.dot(t) * scale
            current_rot = R.dot(current_rot)



            x, y, z = current_pos[0], current_pos[1], current_pos[2]

            
            p = Point()
            

            p.x = x*1.0
            p.y = (z*(1.0))
            p.z = 0;

            sy = math.sqrt(current_rot[0, 0] * current_rot[0, 0] + current_rot[1, 0] * current_rot[1, 0])
            roll = math.atan2(current_rot[2, 1], current_rot[2, 2])

            pitch = math.atan2(-current_rot[2, 0], sy)

            yaw = math.atan2(current_rot[1, 0], current_rot[0, 0])

            roll = math.degrees(roll)
            pitch = math.degrees(pitch)
            yaw = math.degrees(yaw)

            image = cv2.drawKeypoints(image, kps, None)
            
            self.img_pub.publish(br.cv2_to_imgmsg(image, "bgr8"))
	    
            self.points.points.append(p);
	 
            self.point_pub.publish(self.points)

            index = index + 1
            prev_image = image
            prev_keypoint = kp
            self.loop_rate.sleep()





if __name__ == '__main__':
    rospy.init_node("path_demo", anonymous=True)
    my_node = Nodo()
    my_node.start()





