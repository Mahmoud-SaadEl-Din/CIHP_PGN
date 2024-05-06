import os,datetime
from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from werkzeug.utils import secure_filename
import shutil
import cv2
from os.path import join
from inf_pgn import *
import time
from infer_cloth_mask import *
from get_parse_agnostic import get_im_parse_agnostic_original_for_leg, get_img_agnostic_human,get_img_agnostic_human2_for_leg,get_img_agnostic_human3_for_leg, read_pose_parse, read_pose_parse_detectron2
import numpy as np
from PIL import Image
from self_visualized import infer_densepose
from test_generator import infer_hr_viton
from keypoints_detectron2 import *
from flask import send_from_directory
from comman_areas_refining import *

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

@app.route('/serve_images/<path:filename>')
def serve_images(filename):
    return send_from_directory('datasets/HR_VITON_group', filename)

@app.route('/serve_images_stable/<path:filename>')
def serve_images_stable(filename):
    return send_from_directory('samples/unpair', filename)


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


# Define a route for the home page
@app.route('/bulk')
def index_bulk():
    return render_template('index_bulk.html')


@app.route('/route_for_button_1')
def route_for_button_1():
    # Fetch data for button 1 (replace this with your logic)
    start = time.time()
    count = infere_parser("datalake_folder/image","datalake_folder/image-parse-v3")
    end = time.time()
    print(count)
    data = {
        "text": f"processed {count} images in {round((end-start),2)} seconds",
    }
    print(data)
    return jsonify(data)

@app.route('/route_for_button_2')
def route_for_button_2():
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
        agnostic,agnostic_mask = get_im_parse_agnostic_original_for_leg(im_parse, im_parse_np, pose_data)
        out_path = join(out_dir, parse_name)
        agnostic.save(out_path)
        with open(join(out_dir, parse_name_npy), 'wb') as f:
            np.save(f, agnostic_mask)
    
    end = time.time()
    data = {
        "text": f"processed {len(os.listdir(out_dir)) // 2} images in {round((end-start),2)} seconds"
    }
    return jsonify(data)

@app.route('/route_for_button_3')
def route_for_button_3():
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
        agnostic,binary = get_img_agnostic_human3_for_leg(rgb_model, im_parse_np, pose_data)
        out_path = join(out_dir, im_name)
        agnostic.save(out_path)
        binary.save(f"datalake_folder/agnostic-mask/{im_name}")
        
        # send_to_diffusion2(join("datalake_folder", "agnostic-v3.2", im_name),"agnostic_gray")
        # send_to_diffusion2(join("datalake_folder", "agnostic-mask", im_name),"cloth_mask")
        
    end = time.time()
    data = {
        "text": f"processed {len(os.listdir(out_dir))} images in {round((end-start),2)} seconds"
    }
    return jsonify(data)

@app.route('/route_for_button_4')
def route_for_button_4():
    in_dir = "/media/HDD2/VITON/CIHP_PGN/datalake_folder/image"
    out_dir = "/media/HDD2/VITON/CIHP_PGN/datalake_folder/openpose_img"
    json_dir = "/media/HDD2/VITON/CIHP_PGN/datalake_folder/openpose_json"
    # exe_bin_file = "./build/examples/openpose/openpose.bin"
    start = time.time()
    # os.chdir("/media/HDD2/VITON/CIHP_PGN/openpose")
    # cmd_ = f"{exe_bin_file} --image_dir {in_dir} --disable_blending --write_images {out_dir} --hand --write_json {json_dir} --display 0 --net_resolution '64x64'"
    # success = os.system(cmd_)
    pose_dir(in_dir, out_dir, json_dir)
    end = time.time()
    data = {
		"text": f"processed {len(os.listdir(json_dir))} images in {round((end-start),2)} seconds"
	}
    # if success ==0:
    #     os.chdir("/media/HDD2/VITON/CIHP_PGN")
    #     data = {
	# 		"text": f"processed {len(os.listdir(json_dir))} in {round((end-start),2)} seconds"
	# 	}
    return jsonify(data)

@app.route('/route_for_button_5')
def route_for_button_5():
    # Fetch data for button 5 (replace this with your logic)
    start = time.time()
    images_dir = "datalake_folder/image"
    out_dir = "datalake_folder/image-densepose"
    infer_densepose(images_dir,out_dir)
    # for image_name in os.listdir(out_dir):
    #     send_to_diffusion2(join("datalake_folder", "image-densepose", image_name),"densepose")
    end = time.time()
    data = {
        "text": f"processed {len(os.listdir(out_dir))} images in {round((end-start),2)} seconds"
        }
    return jsonify(data)

@app.route('/route_for_button_6')
def route_for_button_6():
    # Fetch data for button 1 (replace this with your logic)
    start = time.time()
    count = infere_cloth_mask("datalake_folder/cloth","datalake_folder/cloth-mask")
    # for image_name in os.listdir("datalake_folder/cloth-mask"):
    #     send_to_diffusion2(join("datalake_folder", "cloth-mask", image_name),"cloth_mask")
    #send_to_diffusion2(img_path, "cloth_mask")
    end = time.time()
    data = {
        "text": f"processed {count} images in {round((end-start),2)} seconds"
    }
    return jsonify(data)

@app.route('/route_for_button_7')
def route_for_button_7():
    # Fetch data for button 1 (replace this with your logic)
    start = time.time()
    out_path = "datasets/HR_VITON_group"
    infer_hr_viton("datalake_folder",out_path,"pairs.txt")
    end = time.time()
    data = {
        "text": f"processed {len(os.listdir(out_path)) // 2} images in {round((end-start),2)} seconds"
    }
    print(data)
    return jsonify(data)

@app.route('/route_for_button_8')
def route_for_button_8():
    # Fetch data for button 1 (replace this with your logic)
    start = time.time()
    infere_parser("datalake/image","datalake/image-parse-v3")
    ##############################################
    in_dir = "/media/HDD2/VITON/CIHP_PGN/datalake/image"
    out_dir = "/media/HDD2/VITON/CIHP_PGN/datalake/openpose_img"
    json_dir = "/media/HDD2/VITON/CIHP_PGN/datalake/openpose_json"
    exe_bin_file = "./build/examples/openpose/openpose.bin"
    os.chdir("/media/HDD2/VITON/CIHP_PGN/openpose")
    cmd_ = f"{exe_bin_file} --image_dir {in_dir} --disable_blending --write_images {out_dir} --hand --write_json {json_dir} --display 0 --net_resolution '64x64'"
    success = os.system(cmd_)
    if success == 1:
        print("true succss")
    #########################################
    os.chdir("/media/HDD2/VITON/CIHP_PGN")
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
    img_path = infere_cloth_mask("/media/HDD2/VITON/CIHP_PGN/datalake/cloth","/media/HDD2/VITON/CIHP_PGN/datalake/cloth-mask")
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

@app.route('/route_for_button_9')
def route_for_button_9():
    # Fetch data for button 1 (replace this with your logic)
    start = time.time()
    if len(os.listdir("samples/unpair")) != len(os.listdir("datalake_folder/image"))*len(os.listdir("datalake_folder/cloth")):
        data = {
            "text": f"NUM of {len(os.listdir('samples/unpair'))} images mismatch number of pairs {len(os.listdir('datalake_folder/image'))*len(os.listdir('datalake_folder/cloth'))}"
        }
        print(data)
        return jsonify(data)


    remove_gray_area(parsed_original_imgs="datalake_folder/image-parse-v3", diffused_imgs="samples/unpair",original_imgs_path="datalake_folder/image",out_path="postprocessing/no_gray")
    infere_parser("postprocessing/no_gray","postprocessing/parsed")
    take_diffused_tshist_to_original("postprocessing/no_gray","postprocessing/parsed","datalake_folder/image","postprocessing/final")
    end = time.time()
    data = {
        "text": f"processed {len(os.listdir('postprocessing/final'))} images in {round((end-start),2)} seconds"
    }
    print(data)
    return jsonify(data)


@app.route('/get_all_images')
def get_all_images():
    # Fetch image paths from your server or database dynamically
    # This is just an example; replace it with your actual logic
    print("recieved the request")
    image_folder = 'datasets/HR_VITON_group'  # Change this to your actual image folder path
    image_paths = [f'/serve_images/{img}' for img in os.listdir(image_folder) if "grid" in img]
    print("recieved the request with images", image_paths)
    return jsonify({'imagePaths': image_paths})

@app.route('/get_all_stable_images')
def get_all__stable_images():
    # Fetch image paths from your server or database dynamically
    # This is just an example; replace it with your actual logic
    print("recieved the request")
    image_folder = 'samples/unpair'  # Change this to your actual image folder path
    image_paths = [f'/serve_images_stable/{img}' for img in os.listdir(image_folder)]
    print("recieved the request with images", image_paths)
    return jsonify({'imagePaths': image_paths})



@app.route('/save_Refresh')
def save_Refresh():
    # Fetch image paths from your server or database dynamically
    # This is just an example; replace it with your actual logic
    main_folder = "datalake_folder"
    
    l = ['image', 'image-parse-v3','image-parse-agnostic-v3.2', 'agnostic-v3.2', 'openpose_img', 'openpose_json','agnostic-v3.2','image-densepose']
    for name in l:
        shutil.rmtree(join(main_folder, name))
        os.mkdir(join(main_folder, name))
    l = ["cloth", "cloth-mask"]
    for name in l:     
        shutil.rmtree(join(main_folder, name))
        os.mkdir(join(main_folder, name))
    # print(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    logged_dir = join("Logged_Trails", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(join(logged_dir, "HR_VITON"))
    os.makedirs(join(logged_dir, "STABLE_VITON"))
    from distutils.dir_util import copy_tree
    copy_tree("/media/HDD2/VITON/CIHP_PGN/datasets/HR_VITON_group", join(logged_dir,"HR_VITON"))
    copy_tree("/media/HDD2/VITON/CIHP_PGN/samples/unpair", join(logged_dir,"STABLE_VITON"))
    l = ["datasets/HR_VITON_group","samples/unpair", "test/test/classified_poses"]
    for name in l:     
        shutil.rmtree(name)
        os.mkdir(name)
    if os.path.exists(join(main_folder,"pairs.txt")):
        os.remove(join(main_folder,"pairs.txt"))
    print("recieved the request to refresh")
    return jsonify({'text': f'Refreshed Successfully successfully'})


# Define a route for handling image uploads
@app.route('/upload_person', methods=['POST'])
def upload_person():
    uploaded_files = request.files.getlist('person_images')
    print(uploaded_files)
    c = 0
    w = 768
    h = 1024
    # Process the uploaded files (you can loop through the files and handle them as needed)
    for file in uploaded_files:
        # Save or process each file
        image_name = file.filename.split("/")[-1]
        file.save('datalake_folder/image/' + image_name)
        img = Image.open(join("datalake_folder", "image", image_name))
        img_resize = img.resize((w, h))
        img_resize.save(join("datalake_folder", "image", image_name))
        # send_to_diffusion2(join("datalake_folder", "image", image_name),"person")
        c+=1

    return jsonify({'text': f'{c} Person images uploaded successfully'})
        

# Define a route for handling image uploads
@app.route('/upload_cloth', methods=['POST'])
def upload_cloth():
    uploaded_files = request.files.getlist('cloth_images')
    c= 0 
    w = 768
    h = 1024
    # Process the uploaded files (you can loop through the files and handle them as needed)
    for file in uploaded_files:
        # Save or process each file
        image_name = file.filename.split("/")[-1]
        file.save('datalake_folder/cloth/' + image_name)
        img = Image.open(join("datalake_folder", "cloth", image_name))
        img_resize = img.resize((w, h))
        img_resize.save(join("datalake_folder", "cloth", image_name))
        # send_to_diffusion2(join("datalake_folder", "cloth", image_name),"cloth")

        c+=1

    return jsonify({'text': f'{c} cloth images uploaded successfully'})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5903)