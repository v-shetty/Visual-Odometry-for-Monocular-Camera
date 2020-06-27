import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

import math
import os
import time




def image_path_left(index):

    image_format_left = '{:06d}.png'


    path = r"C:\Users\yourName\Desktop\qt\VO\src\point_demo\Kitti_Dataset\sequence_00"




    return os.path.join(path, image_format_left.format(index))


def main():


    feature_detector = cv2.FastFeatureDetector_create(threshold=25,
                                                      nonmaxSuppression=True)

    lk_params = dict(winSize=(21, 21),
                     criteria=(cv2.TERM_CRITERIA_EPS |
                               cv2.TERM_CRITERIA_COUNT, 30, 0.03))

    current_pos = np.zeros((3, 1))
    current_rot = np.eye(3)

    traj = np.zeros((800, 600, 3), dtype=np.uint8)

    prev_image = None


    camera_matrix = np.array([[1.87156301e+03, 0.0, 9.55327397e+02],
                              [0.0, 1.87098313e+03, 5.33056795e+02],
                              [0.0, 0.0, 1.0]])
 
    for index in range(0,4980):
    
        print(index)
        # load image
        image = cv2.imread(image_path_left(index))



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

        sy = math.sqrt(current_rot[0, 0] * current_rot[0, 0] + current_rot[1, 0] * current_rot[1, 0])
        roll = math.atan2(current_rot[2, 1], current_rot[2, 2])

        pitch = math.atan2(-current_rot[2, 0], sy)

        yaw = math.atan2(current_rot[1, 0], current_rot[0, 0])

        roll = math.degrees(roll)
        pitch = math.degrees(pitch)
        yaw = math.degrees(yaw)

      
        draw_x, draw_y = int(x) + 290, ((-1) * (int(z))) + 590
        cv2.circle(traj, (draw_x, draw_y), 1, (0, 0, 255), 2)
        cv2.rectangle(traj, (10, 20), (600, 60), (0, 0, 0), -1)

        text = "Coordinates: x=%2fu y=%2fu z=%2fu" % (x, y, z)

        textRot = "rotation: roll=%2fDeg pitch=%2fDeg yaw=%2fDeg" % (roll, pitch, yaw)
        
        #print(textRot)
        cv2.putText(traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
        cv2.putText(traj, textRot, (20, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

        cv2.imshow('Trajectory', traj)

        # position_axes.scatter(current_pos[0][0], current_pos[2][0])
        plt.pause(.01)

        img = cv2.drawKeypoints(image, kps, None)
        cv2.imshow('feature', img)
        cv2.waitKey(1)
        filename = 'savedImage.jpg'
        cv2.imwrite(filename, traj)

        prev_image = image
        prev_keypoint = kp
        #time.sleep(1)



if __name__ == "__main__":
    main()
