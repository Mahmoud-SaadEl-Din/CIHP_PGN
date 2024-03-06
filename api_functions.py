import os
from os.path import join
import numpy as np
from PIL import Image
import time
import requests

from get_parse_agnostic import get_im_parse_agnostic_original,get_img_agnostic_human2, read_pose_parse_detectron2
from self_visualized import infer_densepose
from keypoints_detectron2 import pose_dir

def send_to_diffusion2(image_path, txt):
    # URL of the GPU server where you'll upload the image
    gpu_server_url = 'http://62.67.51.161:5000/upload'  # Replace with your GPU server's URL
    file_name = image_path.split("/")[-1]

    try:
        with open(image_path, 'rb') as image_file:
            files = {
                'image': image_file,
            }
            data = {
                'type': txt,
                'name': file_name
            }

            response = requests.post(gpu_server_url, files=files, data=data)

            if response.status_code == 200:
                print("Files successfully uploaded to GPU server")
                return response.text  # Return the response if required
            else:
                print("Failed to upload files to GPU server")
                return None
    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
        return None



def mask_agnostic():
    # Fetch data for button 1 (replace this with your logic)
    start = time.time()
    images_dir = "datalake_folder/image"
    out_dir = join("datalake_folder","image-parse-agnostic-v3.2")
    for im_name in os.listdir(images_dir):
        print(im_name)
        im_parse, im_parse_np, pose_data, parse_name, parse_name_npy = read_pose_parse_detectron2("datalake_folder",im_name)
        if im_parse ==False:
            print(f"{im_name} ==> OpenPose Json file is not exist")
            continue
        agnostic,agnostic_mask = get_im_parse_agnostic_original(im_parse, im_parse_np, pose_data)
        out_path = join(out_dir, parse_name)
        agnostic.save(out_path)
        with open(join(out_dir, parse_name_npy), 'wb') as f:
            np.save(f, agnostic_mask)
    
    end = time.time()
    data = {
        "text": f"processed {len(os.listdir(out_dir)) // 2} images in {round((end-start),2)} seconds"
    }
    return data

def gray_agnostic():
    # Fetch data for button 1 (replace this with your logic)
    start = time.time()
    images_dir = "datalake_folder/image"
    out_dir = join("datalake_folder","agnostic-v3.2")
    for im_name in os.listdir(images_dir):
        rgb_model = Image.open(join("datalake_folder","image", im_name))
        im_parse, im_parse_np, pose_data, parse_name, parse_name_npy = read_pose_parse_detectron2("datalake_folder",im_name)
        if im_parse ==False:
            print(f"{im_name} ==> OpenPose Json file is not exist")
            continue
        agnostic,binary = get_img_agnostic_human2(rgb_model, im_parse_np, pose_data)
        out_path = join(out_dir, im_name)
        agnostic.save(out_path)
        print(type(binary))
        binary.save(f"datalake_folder/agnostic-mask/{im_name}")
        
        send_to_diffusion2(join("datalake_folder", "agnostic-v3.2", im_name),"agnostic_gray")
        send_to_diffusion2(join("datalake_folder", "agnostic-mask", im_name),"cloth_mask")
        
    end = time.time()
    data = {
        "text": f"processed {len(os.listdir(out_dir))} images in {round((end-start),2)} seconds"
    }
    return data

def detectron_poses():
    in_dir = "/root/diffusion_root/CIHP_PGN/datalake_folder/image"
    out_dir = "/root/diffusion_root/CIHP_PGN/datalake_folder/openpose_img"
    json_dir = "/root/diffusion_root/CIHP_PGN/datalake_folder/openpose_json"
    start = time.time()
    pose_dir(in_dir, out_dir, json_dir)
    end = time.time()
    data = {
		"text": f"processed {len(os.listdir(json_dir))} images in {round((end-start),2)} seconds"
	}

    return data

def detectron_densepose():
    # Fetch data for button 5 (replace this with your logic)
    start = time.time()
    images_dir = "datalake_folder/image"
    out_dir = "datalake_folder/image-densepose"
    infer_densepose(images_dir,out_dir)
    for image_name in os.listdir(out_dir):
        send_to_diffusion2(join("datalake_folder", "image-densepose", image_name),"densepose")
    end = time.time()
    data = {
        "text": f"processed {len(os.listdir(out_dir))} images in {round((end-start),2)} seconds"
        }
    return data

