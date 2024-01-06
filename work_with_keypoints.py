import os
import json
from os.path import join
import numpy as np
import cv2

key_points_file = "/root/diffusion_root/CIHP_PGN/datalake_folder/openpose_json/t22_keypoints.json"
image = cv2.imread("/root/diffusion_root/CIHP_PGN/datalake_folder/image/t22.png")
def read_pose_parse(image):
    with open(key_points_file, 'r') as f:
        pose_label = json.load(f)
        pose_data = pose_label['people'][0]['pose_keypoints_2d']
        pose_data = np.array(pose_data)
        print("herre1",pose_data)
        pose_data = pose_data.reshape((-1, 3))[:, :2]
        print("herre2",pose_data)
        print(pose_data.shape)
        for i, pnt in enumerate(pose_data):
            image = cv2.circle(image, (int(pnt[0]),int(pnt[1])), radius=1, color=(0, 0, 255), thickness=-1)
            image = cv2.putText(image, f'{i}', (int(pnt[0])-5,int(pnt[1])-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
            cv2.imwrite("key_points_openpose.png", image)
            # input()


read_pose_parse(image)