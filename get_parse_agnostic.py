import json
from os import path as osp
import os

import numpy as np
from PIL import Image, ImageDraw

import argparse
import cv2
from tqdm import tqdm

# From openpose:   2 5 6 7 3 4 .. 12 9 
# from detectron2: 6 5 7 9 8 10 .. 11 12

def get_im_parse_agnostic_original(im_parse,parse_array, pose_data):
    # parseArray is 2D mask with classes index
    w, h = im_parse.size

    parse_upper = ((parse_array == 5).astype(np.float32) +
                    (parse_array == 6).astype(np.float32) +
                    (parse_array == 7).astype(np.float32))
    parse_neck = (parse_array == 10).astype(np.float32)

    agnostic_mask = parse_array.copy()
    agnostic_mask[parse_array==5] = 0
    agnostic_mask[parse_array==6] = 0
    agnostic_mask[parse_array==7] = 0
    agnostic_mask[parse_array==10] = 0

    r = 10
    agnostic = im_parse.copy()

    # mask arms
    for parse_id, pose_ids in [(14, [6, 5, 7, 9]), (15, [5, 6, 8, 10])]:
        mask_arm = Image.new('L', (w, h), 'black')
        # print(mask_arm, type(mask_arm))
        mask_arm_draw = ImageDraw.Draw(mask_arm)
        i_prev = pose_ids[0]
        for i in pose_ids[1:]:
            if (pose_data[i_prev, 0] == 0.0 and pose_data[i_prev, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            mask_arm_draw.line([tuple(pose_data[j]) for j in [i_prev, i]], 'white', width=r*10)
            pointx, pointy = pose_data[i]
            radius =r*2 if i == pose_ids[-1] else r*10
            mask_arm_draw.ellipse((pointx-radius, pointy-radius, pointx+radius, pointy+radius), 'white', 'white')
            i_prev = i

        parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

    # mask torso & neck
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_neck * 255), 'L'))

    return agnostic,agnostic_mask


def get_img_agnostic_human(img, parse, pose_data):
    parse_array = parse
    parse_head = ((parse_array == 4).astype(np.float32) +
                    (parse_array == 13).astype(np.float32))
    parse_lower = ((parse_array == 9).astype(np.float32) +
                    (parse_array == 12).astype(np.float32) +
                    (parse_array == 16).astype(np.float32) +
                    (parse_array == 17).astype(np.float32) +
                    (parse_array == 18).astype(np.float32) +
                    (parse_array == 19).astype(np.float32))

    
    agnostic = img.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)

    length_a = np.linalg.norm(pose_data[5] - pose_data[6])
    length_b = np.linalg.norm(pose_data[11] - pose_data[12])
    point = (pose_data[12] + pose_data[11]) / 2
    pose_data[12] = point + (pose_data[12] - point) / length_b * length_a
    pose_data[11] = point + (pose_data[11] - point) / length_b * length_a
    r = int(length_a / 16) + 1
    
    # mask arms
    # agnostic_draw.line([tuple(pose_data[i]) for i in [6, 5]], 'gray', width=r*7)
    for i in [6, 5]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
    for i in [8, 10, 7, 9]:
        if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
            continue
        agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*6)
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*2, pointy-r*2, pointx+r*2, pointy+r*2), 'gray', 'gray')

    # mask torso
    for i in [12, 11]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
    agnostic_draw.line([tuple(pose_data[i]) for i in [6, 12]], 'gray', width=r*10)
    agnostic_draw.line([tuple(pose_data[i]) for i in [5, 11]], 'gray', width=r*10)
    agnostic_draw.line([tuple(pose_data[i]) for i in [12, 11]], 'gray', width=r*12)
    agnostic_draw.polygon([tuple(pose_data[i]) for i in [6, 5, 11, 12]], 'gray', 'gray')

    # mask neck
    pointx, pointy = pose_data[1]
    pointx, pointy = (pose_data[5] + pose_data[6]) / 2
    agnostic_draw.rectangle((pointx-r*8, pointy-r*8, pointx+r*8, pointy+r*8), 'gray', 'gray')
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))

    return agnostic


def read_pose_parse(root_path, im_name):
    ext = "."+im_name.split(".")[-1]
    # load pose image
    pose_name = im_name.replace(ext, '_keypoints.json')

    pose_data = None
    try:
        with open(osp.join(root_path,"openpose_json",pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints_2d']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))[:, :2]
    except:
        return False,False,False,False,False
     

    # load parsing image
    parse_name = im_name.replace(ext, '_vis.png')
    parse_name_npy = im_name.replace(ext, '_vis2.npy')
    im_parse = Image.open(osp.join(root_path, 'image-parse-v3', parse_name))
    with open(osp.join(root_path, 'image-parse-v3', parse_name_npy), 'rb') as f:
        im_parse_np = np.load(f)

    return im_parse, im_parse_np, pose_data, parse_name, parse_name_npy


def read_pose_parse_detectron2(root_path, im_name):
    ext = "."+im_name.split(".")[-1]
    # load pose image
    pose_name = im_name.replace(ext, '.npy')

    pose_data = None
    try:
        with open(osp.join(root_path,"openpose_json",pose_name), 'rb') as f:
            pose_data = np.load(f)
            pose_data = pose_data.reshape((-1, 3))[:, :2]
    except:
        return False,False,False,False,False
     

    # load parsing image
    parse_name = im_name.replace(ext, '_vis.png')
    parse_name_npy = im_name.replace(ext, '_vis2.npy')
    # print(parse_name, parse_name_npy, pose_data)
    im_parse = Image.open(osp.join(root_path, 'image-parse-v3', parse_name))
    with open(osp.join(root_path, 'image-parse-v3', parse_name_npy), 'rb') as f:
        im_parse_np = np.load(f)

    return im_parse, im_parse_np, pose_data, parse_name, parse_name_npy



# im_name = "t12.png"
# rgb_model = Image.open("/root/diffusion_root/CIHP_PGN/datalake_folder/image/t12.png")
# im_parse, im_parse_np, pose_data, parse_name, parse_name_npy = read_pose_parse_detectron2("datalake_folder",im_name)
# if im_parse ==False:
#     print(f"{im_name} ==> OpenPose Json file is not exist")
# agnostic = get_img_agnostic_human(rgb_model, im_parse_np, pose_data)
# out_path = "test.png"
# agnostic.save(out_path)
# agnostic,agnostic_mask = get_im_parse_agnostic_original(im_parse, im_parse_np, pose_data)
# out_path = "test2.png"
# agnostic.save(out_path)   
