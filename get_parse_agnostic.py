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
    # agnostic_draw.line([tuple(pose_data[i]) for i in [6, 5]], color, width=r*7)
    color = (128,128,128)
    for i in [6, 5]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), color, color)
    for i in [8, 10, 7, 9]:
        if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
            continue
        agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], color, width=r*6)
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*2, pointy-r*2, pointx+r*2, pointy+r*2), color, color)

    # mask torso
    for i in [12, 11]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), color, color)
    agnostic_draw.line([tuple(pose_data[i]) for i in [6, 12]], color, width=r*10)
    agnostic_draw.line([tuple(pose_data[i]) for i in [5, 11]], color, width=r*10)
    agnostic_draw.line([tuple(pose_data[i]) for i in [12, 11]], color, width=r*12)
    agnostic_draw.polygon([tuple(pose_data[i]) for i in [6, 5, 11, 12]], color, color)

    # mask neck
    pointx, pointy = pose_data[1]
    pointx, pointy = (pose_data[5] + pose_data[6]) / 2
    agnostic_draw.rectangle((pointx-r*8, pointy-r*8, pointx+r*8, pointy+r*8), color, color)
    
    
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))

    return agnostic


def get_im_parse_agnostic_original_for_leg(im_parse,parse_array, pose_data):
    # parseArray is 2D mask with classes index
    w, h = im_parse.size

    # 16, 17: Right/left leg
    # 8,9,12: Socks, pants, skirt 
    parse_leg = ((parse_array == 17).astype(np.float32) +
                    (parse_array == 16).astype(np.float32)) 
    parse_lower_cloth = ((parse_array == 8).astype(np.float32)+
                        (parse_array == 9).astype(np.float32)+ 
                        (parse_array == 12).astype(np.float32))

    agnostic_mask = parse_array.copy()
    agnostic_mask[parse_array==17] = 0
    agnostic_mask[parse_array==16] = 0
    agnostic_mask[parse_array==8] = 0
    agnostic_mask[parse_array==9] = 0
    agnostic_mask[parse_array==12] = 0

    r = 10
    agnostic = im_parse.copy()
    # mask torso & neck
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_leg * 255), 'L'))
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_lower_cloth * 255), 'L'))

    return agnostic,agnostic_mask


def get_img_agnostic_human2(img, parse, pose_data, color=(128, 128, 128),color_mask=(255,255,255)):
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

    # Create a binary mask to track filled areas with RGB channels
    binary_mask = Image.new("RGB", img.size, (0, 0, 0))
    white_img = Image.new("RGB", img.size, (0, 0, 0))

    binary_mask_draw = ImageDraw.Draw(binary_mask)

    
    
    # agnostic_draw.ellipse((pose_data[6][0],pose_data[6][1], pose_data[11][0],pose_data[11][1]), color, color)
    # agnostic_draw.ellipse((pose_data[5][0],pose_data[5][1], pose_data[11][0],pose_data[11][1]), color, color)
    # agnostic_draw.ellipse((pose_data[6][0],pose_data[6][1], pose_data[12][0],pose_data[12][1]), color, color)
    # agnostic_draw.ellipse((pose_data[5][0],pose_data[5][1], pose_data[12][0],pose_data[12][1]), color, color)
    # print(pose_data[5])
    pairs = [[5,6],[7,8],[9,10],[5,7],[7,9],[6,8],[8,10]]
    for i,j in pairs:
        shape = [tuple(pose_data[i]),tuple(pose_data[j])]
        agnostic_draw.line(shape, color, width=110)
        binary_mask_draw.line(shape, color_mask, width=110)


    length_a = np.linalg.norm(pose_data[5] - pose_data[6])
    length_b = np.linalg.norm(pose_data[11] - pose_data[12])
    point = (pose_data[12] + pose_data[11]) / 2
    pose_data[12] = point + (pose_data[12] - point) / length_b * length_a
    pose_data[11] = point + (pose_data[11] - point) / length_b * length_a
    r = int(length_a / 16) + 1

    # mask arms
    for i in [6, 5]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx - r * 5, pointy - r * 5, pointx + r * 5, pointy + r * 5), color, color)
        binary_mask_draw.ellipse((pointx - r * 5, pointy - r * 5, pointx + r * 5, pointy + r * 5), color_mask, color_mask)

    for i in [8, 10, 7, 9]:
        if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (
                pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
            continue
        agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], color, width=r * 6)
        binary_mask_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], color_mask, width=r * 6)
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx - r * 2, pointy - r * 2, pointx + r * 2, pointy + r * 2), color, color)
        binary_mask_draw.ellipse((pointx - r * 2, pointy - r * 2, pointx + r * 2, pointy + r * 2), color_mask, color_mask)

    # mask torso
    for i in [12, 11]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx - r * 3, pointy - r * 6, pointx + r * 3, pointy + r * 6), color, color)
        binary_mask_draw.ellipse((pointx - r * 3, pointy - r * 6, pointx + r * 3, pointy + r * 6), color_mask, color_mask)
    agnostic_draw.line([tuple(pose_data[i]) for i in [6, 12]], color, width=r * 10)
    agnostic_draw.line([tuple(pose_data[i]) for i in [5, 11]], color, width=r * 10)
    agnostic_draw.line([tuple(pose_data[i]) for i in [12, 11]], color, width=r * 12)
    agnostic_draw.polygon([tuple(pose_data[i]) for i in [6, 5, 11, 12]], color, color)
    binary_mask_draw.line([tuple(pose_data[i]) for i in [6, 12]], color_mask, r * 10)
    binary_mask_draw.line([tuple(pose_data[i]) for i in [5, 11]], color_mask, r * 10)
    binary_mask_draw.line([tuple(pose_data[i]) for i in [12, 11]], color_mask, r * 12)
    binary_mask_draw.polygon([tuple(pose_data[i]) for i in [6, 5, 11, 12]], color_mask, color_mask)

    # mask neck
    pointx, pointy = (pose_data[5] + pose_data[6]) / 2
    agnostic_draw.rectangle((pointx-r*8, pointy-r*8, pointx+r*8, pointy+r*8), color, color)
    binary_mask_draw.rectangle((pointx - r * 8, pointy - r * 8, pointx + r * 8, pointy + r * 8), color_mask, color_mask)

    # Paste the original image based on masks
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))
    binary_mask.paste(white_img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    binary_mask.paste(white_img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))
    return agnostic, binary_mask


def get_img_agnostic_human2_for_leg(img, parse, pose_data, color=(128, 128, 128),color_mask=(255,255,255)):
    parse_array = parse
    # 16, 17: Right/left leg
    # 8,9,12: Socks, pants, skirt 
    w, h = img.size
    
    parse_upper = ((parse_array == 5).astype(np.float32) +
                    (parse_array == 6).astype(np.float32) +
                    (parse_array == 19).astype(np.float32) +
                    (parse_array == 18).astype(np.float32) +
                    (parse_array == 14).astype(np.float32) +
                    (parse_array == 15).astype(np.float32) +
                    (parse_array == 0).astype(np.float32) +
                    (parse_array == 10).astype(np.float32) +
                    (parse_array == 7).astype(np.float32))

    agnostic = img.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)

    # Create a binary mask to track filled areas with RGB channels
    binary_mask = Image.new("RGB", img.size, (0, 0, 0))
    white_img = Image.new("RGB", img.size, (0, 0, 0))

    binary_mask_draw = ImageDraw.Draw(binary_mask)

    # Leg points. Right Leg (12,14,16) . Left Leg (11,13,15)
    beginning_of_lower = pose_data[12][1] - 50
    
    x,y = 0,beginning_of_lower
    x1,y1 = w,h
    agnostic_draw.rectangle(((x,y),(x1,y1)),color)
    binary_mask_draw.rectangle(((x,y),(x1,y1)),color_mask)

    
    # print(pose_data[5])
    # pairs = [[11,12],[11,13],[13,15],[12,14],[14,16]]
    # for i,j in pairs:
    #     shape = [tuple(pose_data[i]),tuple(pose_data[j])]
    #     agnostic_draw.line(shape, color, width=140)
    #     binary_mask_draw.line(shape, color_mask, width=140)

    # Paste the original image based on masks
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
    binary_mask.paste(white_img, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
    return agnostic, binary_mask

def get_img_agnostic_human3_for_leg(img, parse, pose_data, color=(128, 128, 128),color_mask=(255,255,255)):
    parse_array = parse
    # 16, 17: Right/left leg
    # 8,9,12: Socks, pants, skirt 
    w, h = img.size
    
    parse_upper = ((parse_array == 5).astype(np.float32) +
                    (parse_array == 6).astype(np.float32) +
                    (parse_array == 19).astype(np.float32) +
                    (parse_array == 18).astype(np.float32) +
                    (parse_array == 14).astype(np.float32) +
                    (parse_array == 15).astype(np.float32) +
                    (parse_array == 10).astype(np.float32) +
                    (parse_array == 7).astype(np.float32))

    agnostic = img.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)

    # Create a binary mask to track filled areas with RGB channels
    binary_mask = Image.new("RGB", img.size, (0, 0, 0))
    white_img = Image.new("RGB", img.size, (0, 0, 0))

    binary_mask_draw = ImageDraw.Draw(binary_mask)

    # Leg points. Right Leg (12,14,16) . Left Leg (11,13,15)
    
    y1 = int(pose_data[12][1])
    y2 = int(pose_data[15][1])
    
    pose_data[12][0] = pose_data[12][0] - 50
    pose_data[12][1] = pose_data[12][1] - 50
    pose_data[15][0] = pose_data[15][0] + 50
    pose_data[15][1] = pose_data[15][1] + 50
    if y1 > y2:
        print('smaller')
        pose_data[15][1] = h
        
    agnostic_draw.rectangle((tuple(pose_data[12]),tuple(pose_data[15])),color)
    
    binary_mask_draw.rectangle((tuple(pose_data[12]),tuple(pose_data[15])),color_mask)

    
    # Paste the original image based on masks
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
    binary_mask.paste(white_img, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
    return agnostic, binary_mask


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
