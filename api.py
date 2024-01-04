import os
from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from werkzeug.utils import secure_filename
import shutil
import cv2
from os.path import join
from inf_pgn import *
import time
from infer_cloth_mask import *
from get_parse_agnostic import get_im_parse_agnostic_original, get_img_agnostic_human, read_pose_parse
import numpy as np
from PIL import Image
from self_visualized import infer_densepose
from test_generator import infer_hr_viton


app = Flask(__name__)


app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # for many images


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


import requests


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

@app.route('/server_2_call', methods=['POST'])
def upload_image():
    uploaded_image = request.files['image']
    uploaded_file_name = request.form['path']
    
    uploaded_image.save(uploaded_file_name)  # Save the image to a file

    # Return a response if needed (optional)
    return 'Image received and processed on GPU server'

# Define a route for the home page
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/route_for_button_1')
def route_for_button_1():
    # Fetch data for button 1 (replace this with your logic)
    start = time.time()
    img_path = infere_parser("datalake/image","datalake/image-parse-v3")
    print(img_path)
    np = img_path.split(".")
    path_np = np[0]+ "2.npy"
    #send_to_diffusion2(img_path,"parse_orange")
    #send_to_diffusion2(path_np,"parse_orange")
    end = time.time()
    parse_location = f"static/images/hmp_{img_path.split('/')[-1]}"
    shutil.copy(img_path, parse_location)
    cv2.imwrite(parse_location, image_resize(cv2.imread(parse_location), height=256))
    data = {
        "text": f"processed in {round((end-start),2)} seconds",
        "image": parse_location
    }
    print(data)
    return jsonify(data)

@app.route('/route_for_button_2')
def route_for_button_2():
    # Fetch data for button 1 (replace this with your logic)
    start = time.time()
    f = open("name.txt", "r")
    im_name = f.read()
    print(im_name)
    im_parse, im_parse_np, pose_data, parse_name, parse_name_npy = read_pose_parse("datalake",im_name)
    if im_parse ==False:
        data = {
            "text": "OpenPose Json file is not exist",
            "image": ""
        }
        return jsonify(data)
    agnostic,agnostic_mask = get_im_parse_agnostic_original(im_parse, im_parse_np, pose_data)
    out_path = join("datalake","image-parse-agnostic-v3.2", parse_name)
    agnostic.save(out_path)
    #send_to_diffusion2(out_path, "agnostic_black")
    with open(join("datalake","image-parse-agnostic-v3.2", parse_name_npy), 'wb') as f:
        np.save(f, agnostic_mask)
    #send_to_diffusion2(join("datalake","image-parse-agnostic-v3.2", parse_name_npy), "agnostic_black")    
    end = time.time()
    parse_location = f"static/images/agp_{im_name}"
    shutil.copy(out_path, parse_location)
    cv2.imwrite(parse_location, image_resize(cv2.imread(parse_location), height=256))
    data = {
        "text": f"processed in {round((end-start),2)} seconds",
        "image": parse_location
    }
    return jsonify(data)

@app.route('/route_for_button_3')
def route_for_button_3():
    # Fetch data for button 1 (replace this with your logic)
    start = time.time()
    f = open("name.txt", "r")
    im_name = f.read()
    rgb_model = Image.open(join("datalake","image", im_name))
    im_parse, im_parse_np, pose_data, parse_name, parse_name_npy = read_pose_parse("datalake",im_name)
    if im_parse ==False:
        data = {
            "text": "OpenPose Json file is not exist",
            "image": ""
        }
        return jsonify(data)
    agnostic = get_img_agnostic_human(rgb_model, im_parse_np, pose_data)
    out_path = join("datalake","agnostic-v3.2", im_name)
    agnostic.save(out_path)
    #send_to_diffusion2(out_path, "agnostic_gray")
    end = time.time()
    parse_location = f"static/images/agh_{im_name}"
    shutil.copy(out_path, parse_location)
    cv2.imwrite(parse_location, image_resize(cv2.imread(parse_location), height=256))
    data = {
        "text": f"processed in {round((end-start),2)} seconds",
        "image": parse_location
    }
    return jsonify(data)

@app.route('/route_for_button_4')
def route_for_button_4():
    in_dir = "/root/diffusion_root/CIHP_PGN/datalake/image"
    out_dir = "/root/diffusion_root/CIHP_PGN/datalake/openpose_img"
    json_dir = "/root/diffusion_root/CIHP_PGN/datalake/openpose_json"
    exe_bin_file = "./build/examples/openpose/openpose.bin"
    start = time.time()
    os.chdir("/root/diffusion_root/CIHP_PGN/openpose")
    cmd_ = f"{exe_bin_file} --image_dir {in_dir} --disable_blending --write_images {out_dir} --hand --write_json {json_dir} --display 0 --net_resolution '64x64'"
    success = os.system(cmd_)
    data = {
		"text": "Faild",
		"image": ""
	}
    if success ==0:
        end = time.time()
        os.chdir("/root/diffusion_root/CIHP_PGN")
        f = open("name.txt", "r")
        im_name = f.read()
        ext="."+im_name.split(".")[-1]
        im_name_datalake = im_name.replace(ext,"_rendered.png")
        im_name_datalake_json = im_name.replace(ext,"_keypoints.json")
        parse_location = f"static/images/pose_{im_name}"
        shutil.copy(join(out_dir,im_name_datalake), parse_location)
        #send_to_diffusion2(join(out_dir,im_name_datalake),"openpose_render")
        #send_to_diffusion2(join(json_dir,im_name_datalake_json),"openpose_json")
        cv2.imwrite(parse_location, image_resize(cv2.imread(parse_location), height=256))
        data = {
			"text": f"processed in {round((end-start),2)} seconds",
			"image": parse_location
		}
    return jsonify(data)

@app.route('/route_for_button_5')
def route_for_button_5():
    # Fetch data for button 5 (replace this with your logic)
    start = time.time()
    infer_densepose("datalake/image","datalake/image-densepose")
    end = time.time()
    f = open("name.txt", "r")
    im_name = f.read()
    parse_location = f"static/images/dense_{im_name}"
    shutil.copy(join("datalake","image-densepose", im_name), parse_location)
    cv2.imwrite(parse_location, image_resize(cv2.imread(parse_location), height=256))
    data = {
        "text": f"processed in {round((end-start),2)} seconds",
        "image": parse_location
    }
    print(data)
    return jsonify(data)

@app.route('/route_for_button_6')
def route_for_button_6():
    # Fetch data for button 1 (replace this with your logic)
    start = time.time()
    img_path = infere_cloth_mask("/root/diffusion_root/CIHP_PGN/datalake/cloth","/root/diffusion_root/CIHP_PGN/datalake/cloth-mask")
    #send_to_diffusion2(img_path, "cloth_mask")
    end = time.time()
    parse_location = f"static/images/cm_{img_path.split('/')[-1]}"
    shutil.copy(img_path, parse_location)
    cv2.imwrite(parse_location, image_resize(cv2.imread(parse_location), height=256))
    data = {
        "text": f"processed in {round((end-start),2)} seconds",
        "image": parse_location
    }
    print(data)
    return jsonify(data)

@app.route('/route_for_button_7')
def route_for_button_7():
    # Fetch data for button 1 (replace this with your logic)
    start = time.time()
    out_path = "datasets/HR_VITON_outs"
    infer_hr_viton("datalake",out_path,"pairs.txt")
    end = time.time()
    paired = os.listdir(out_path)
    im_name = paired[0] if "grid" in paired[1] else paired[1]
    parse_location = f"static/images/paired_{im_name}"
    shutil.copy(join(out_path, im_name), parse_location)
    cv2.imwrite(parse_location, image_resize(cv2.imread(parse_location), height=256))
    data = {
        "text": f"processed in {round((end-start),2)} seconds",
        "image": parse_location
    }
    print(data)
    return jsonify(data)

@app.route('/route_for_button_8')
def route_for_button_8():
    # Fetch data for button 1 (replace this with your logic)
    start = time.time()
    infere_parser("datalake/image","datalake/image-parse-v3")
    ##############################################
    in_dir = "/root/diffusion_root/CIHP_PGN/datalake/image"
    out_dir = "/root/diffusion_root/CIHP_PGN/datalake/openpose_img"
    json_dir = "/root/diffusion_root/CIHP_PGN/datalake/openpose_json"
    exe_bin_file = "./build/examples/openpose/openpose.bin"
    os.chdir("/root/diffusion_root/CIHP_PGN/openpose")
    cmd_ = f"{exe_bin_file} --image_dir {in_dir} --disable_blending --write_images {out_dir} --hand --write_json {json_dir} --display 0 --net_resolution '64x64'"
    success = os.system(cmd_)
    if success == 1:
        print("true succss")
    #########################################
    os.chdir("/root/diffusion_root/CIHP_PGN")
    f = open("name.txt", "r")
    im_name = f.read()
    print(im_name)
    im_parse, im_parse_np, pose_data, parse_name, parse_name_npy = read_pose_parse("datalake",im_name)
    agnostic,agnostic_mask = get_im_parse_agnostic_original(im_parse, im_parse_np, pose_data)
    out_path = join("datalake","image-parse-agnostic-v3.2", parse_name)
    agnostic.save(out_path)

    with open(join("datalake","image-parse-agnostic-v3.2", parse_name_npy), 'wb') as f:
        np.save(f, agnostic_mask)
    ########################################
    rgb_model = Image.open(join("datalake","image", im_name))
    agnostic = get_img_agnostic_human(rgb_model, im_parse_np, pose_data)
    out_path = join("datalake","agnostic-v3.2", im_name)
    agnostic.save(out_path)
    ############################################
    infer_densepose("datalake/image","datalake/image-densepose")
    ############################################
    img_path = infere_cloth_mask("/root/diffusion_root/CIHP_PGN/datalake/cloth","/root/diffusion_root/CIHP_PGN/datalake/cloth-mask")
    ############################################
    out_path = "datasets/HR_VITON_outs"
    infer_hr_viton("datalake",out_path,"pairs.txt")
    paired = os.listdir(out_path)
    im_name = paired[0] if "grid" in paired[1] else paired[1]
    parse_location = f"static/images/paired_{im_name}"
    shutil.copy(join(out_path, im_name), parse_location)
    cv2.imwrite(parse_location, image_resize(cv2.imread(parse_location), height=256))
    end = time.time()
    data = {
        "text": f"processed in {round((end-start),2)} seconds",
        "image": parse_location
    }
    print(data)
    return jsonify(data)

# Define a route for handling image uploads
@app.route('/upload_person', methods=['POST'])
def upload_person():
    w = 768
    h = 1024
    l = ['image', 'image-parse-v3','image-parse-agnostic-v3.2', 'agnostic-v3.2', 'openpose_img', 'openpose_json','agnostic-v3.2','image-densepose']
    for name in l:
        shutil.rmtree(join("datalake", name))
        os.mkdir(join("datalake", name))
    start = time.time()
    # Handle the uploaded file here
    uploaded_file = request.files['file_p']
    img_name = uploaded_file.filename
    uploaded_file.save(join("datalake", "image", img_name))

    img = Image.open(join("datalake", "image", img_name))
    img_resize = img.resize((w, h))
    img_resize.save(join("datalake", "image", img_name))
    #send_to_diffusion2(join("datalake", "image", img_name),"person")
    end = time.time()
    parse_location = f"static/images/paired_{img_name}"
    shutil.copy(join("datalake", "image", img_name), parse_location)
    cv2.imwrite(parse_location, image_resize(cv2.imread(parse_location), height=256))
    data = {
		"text": f"processed in {round((end-start),2)} seconds",
		"image": parse_location
	}
    print(data)
    os.remove("name.txt")
    f = open("name.txt", "a")
    f.write(img_name)
    f.close()
    #send_to_diffusion2("name.txt","names")
    # Return a response to the front end
    return jsonify(data)
        

# Define a route for handling image uploads
@app.route('/upload_cloth', methods=['POST'])
def upload_cloth():
    w = 768
    h = 1024
    l = ["cloth", "cloth-mask"]
    for name in l:     
        shutil.rmtree(join("datalake", name))
        os.mkdir(join("datalake", name))
    l = ["HR_VITON_outs"]
    for name in l:     
        shutil.rmtree(join("datasets", name))
        os.mkdir(join("datasets", name))
    if os.path.exists(join("datalake","pairs.txt")):
        os.remove(join("datalake","pairs.txt"))
    start = time.time()
    # Handle the uploaded file here
    uploaded_file = request.files['file_c']
    img_name = uploaded_file.filename
    uploaded_file.save(join("datalake", "cloth", img_name))
    img = Image.open(join("datalake", "cloth", img_name))
    img_resize = img.resize((w, h))
    img_resize.save(join("datalake", "cloth", img_name))
    #send_to_diffusion2(join("datalake", "cloth", img_name),"cloth")
    end = time.time()
    parse_location = f"static/images/unpaired_{img_name}"
    shutil.copy(join("datalake", "cloth", img_name), parse_location)
    cv2.imwrite(parse_location, image_resize(cv2.imread(parse_location), height=256))
    data = {
		"text": f"processed in {round((end-start),2)} seconds",
		"image": parse_location
	}
    # Return a response to the front end
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5000)