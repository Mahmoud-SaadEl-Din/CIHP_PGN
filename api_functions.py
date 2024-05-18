import os
from os.path import join
import numpy as np
from PIL import Image
import time
import requests
import cv2
import pandas as pd
from get_parse_agnostic import get_im_parse_agnostic_original,get_img_agnostic_human2, read_pose_parse_detectron2, get_img_agnostic_human3_for_leg
from self_visualized import infer_densepose
from keypoints_detectron2 import pose_dir
import statsmodels.api as sm 
from sklearn.model_selection import train_test_split
# import gevent
import threading
from tqdm import tqdm

# from gevent import monkey
import requests

# Patch the standard library to use gevent for networking
# monkey.patch_all()

def send_async_request(url):
    # Use requests.get() to send the HTTP GET request asynchronously
    response = requests.get(url)
    # Optionally, handle the response or perform other tasks here
    
    print(f"Request sent to {url}")
    print(f"Request sent to {url}, response status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response data: {response.json()}")

def ask_server2_to_diffuse(working_dir):
    url = f"http://62.67.51.161:5000/run_SV/{working_dir}"
    # greenlet = gevent.spawn(send_async_request, url)
    thread = threading.Thread(target=send_async_request, args=(url,))
    thread.start()
    return "request sent"


# TODO: to be done later, multi-request and resources control 
def run_prerequiste_in_bk(img_id):
    requests.get(f"http://62.67.51.161:5903/prerequiste/{img_id}")

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

def gray_agnostic(root="datalake"):
    # Fetch data for button 1 (replace this with your logic)
    start = time.time()
    images_dir = f"{root}/image"
    out_dir = join(root,"agnostic-v3.2")
    
    for im_name in os.listdir(images_dir):
        rgb_model = Image.open(join(root,"image", im_name))
        im_parse, im_parse_np, pose_data, parse_name, parse_name_npy = read_pose_parse_detectron2(root,im_name)
        if im_parse ==False:
            print(f"{im_name} ==> OpenPose Json file is not exist")
            continue
        agnostic,binary = get_img_agnostic_human2(rgb_model, im_parse_np, pose_data)
        out_path = join(out_dir, im_name)
        agnostic.save(out_path)
        binary.save(f"{root}/agnostic-mask/{im_name}")
    end = time.time()
    data = {
        "text": f"processed {len(os.listdir(out_dir))} images in {round((end-start),2)} seconds"
    }
    return data

def gray_agnostic_pure(in_):
    for im_name in tqdm(os.listdir(join(in_,"images"))):
        rgb_model = Image.open(join(in_, "images",im_name))
        im_parse, im_parse_np, pose_data, parse_name, parse_name_npy = read_pose_parse_detectron2(in_,im_name)
        if im_parse ==False:
            print(f"{im_name} ==> OpenPose Json file is not exist")
            continue
        agnostic,binary = get_img_agnostic_human3_for_leg(rgb_model, im_parse_np, pose_data)
        out_path = join(in_,"agnostic", im_name)
        agnostic.save(out_path)
        # print(type(binary))
        binary.save(f"{in_}/agnostic-mask/{im_name}")
     

def detectron_poses(root="datalake"):
    start = time.time()
    in_dir = join(root, "image")
    out_dir = join(root, "openpose_img")
    json_dir = join(root, "openpose_json")
    pose_dir(in_dir, out_dir, json_dir)
    end = time.time()
    data = {
		"text": f"processed {len(os.listdir(json_dir))} images in {round((end-start),2)} seconds"
	}

    return data

def detectron_densepose(root="datalake"):
    # Fetch data for button 5 (replace this with your logic)
    start = time.time()
    images_dir = f"{root}/image"
    out_dir = f"{root}/image-densepose"
    infer_densepose(images_dir,out_dir)
    end = time.time()
    data = {
        "text": f"processed {len(os.listdir(out_dir))} images in {round((end-start),2)} seconds"
        }
    return data

def normalize_pose(r,image_name):
    results = [f"{image_name}"]
    h, w, channel = cv2.imread(f"{r}/image/{image_name}").shape
    with open(f"{r}/openpose_json/{image_name.split('.')[0]}.npy", 'rb') as f:
        pose_data = np.load(f)
        pose_data = pose_data.reshape((-1, 3))[:, :2]
        for pairs in pose_data:
            norm_x, norm_y = round(pairs[0]/w ,2), round(pairs[1]/h, 2)
            results.extend([norm_x,norm_y])
    
    return results


def poses_classification_with_sm(root='temp_poses'):
    a = []
    for image_name in os.listdir(join(root,"image")):
        l = normalize_pose(root,image_name)
        a.append(l)    
    df = pd.DataFrame(a)
 
    # defining the dependent and independent variables 
    X_train = df

    # Splitting the data into train and test
    X_train_name = X_train[X_train.columns[0]].tolist()
    X_train = X_train.drop(X_train.columns[0], axis=1)
   
    log_reg = sm.regression.linear_model.OLSResults.load("Log_reg.pt")

    yhat = log_reg.predict(X_train) 
    prediction = list(map(round, yhat)) 
    
    return X_train_name, list(yhat), prediction





