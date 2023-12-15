import os
from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from werkzeug.utils import secure_filename
import shutil
import cv2
from os.path import join
from inf_pgn import infere_parser
import time
from infer_cloth_mask import infere_cloth_mask
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


# Define a route for the home page
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/not_revised', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		#print('upload_image filename: ' + filename)
		flash('Image successfully uploaded and displayed below')
		return render_template('upload.html', filename=filename)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)


@app.route('/route_for_button_1')
def route_for_button_1():
    # Fetch data for button 1 (replace this with your logic)
    start = time.time()
    img_path = infere_parser("datalake/image","datalake/image-parse-v3")
    end = time.time()
    parse_location = f"static/images/hmp_{img_path.split('/')[-1]}"
    shutil.copy(img_path, parse_location)
    cv2.imwrite(parse_location, image_resize(cv2.imread(parse_location), height=256))
    data = {
        "text": f"processed in {round(((end-start)/ 60),2)} mintues",
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

    with open(join("datalake","image-parse-agnostic-v3.2", parse_name_npy), 'wb') as f:
        np.save(f, agnostic_mask)
        
    end = time.time()
    parse_location = f"static/images/agp_{im_name}"
    shutil.copy(out_path, parse_location)
    cv2.imwrite(parse_location, image_resize(cv2.imread(parse_location), height=256))
    data = {
        "text": f"processed in {round(((end-start)/ 60),2)} mintues",
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
    end = time.time()
    parse_location = f"static/images/agh_{im_name}"
    shutil.copy(out_path, parse_location)
    cv2.imwrite(parse_location, image_resize(cv2.imread(parse_location), height=256))
    data = {
        "text": f"processed in {round(((end-start)/ 60),2)} mintues",
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
    cmd_ = f"{exe_bin_file} --image_dir {in_dir} --disable_blending --write_images {out_dir} --hand --write_json {json_dir} --display 0"
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
        parse_location = f"static/images/pose_{im_name}"
        shutil.copy(join(out_dir,im_name_datalake), parse_location)
        cv2.imwrite(parse_location, image_resize(cv2.imread(parse_location), height=256))
        data = {
			"text": f"processed in {round(((end-start)/ 60),2)} mintues",
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
        "text": f"processed in {round(((end-start)/ 60),2)} mintues",
        "image": parse_location
    }
    print(data)
    return jsonify(data)

@app.route('/route_for_button_6')
def route_for_button_6():
    # Fetch data for button 1 (replace this with your logic)
    start = time.time()
    img_path = infere_cloth_mask("/root/diffusion_root/CIHP_PGN/datalake/cloth","/root/diffusion_root/CIHP_PGN/datalake/cloth-mask")
    end = time.time()
    parse_location = f"static/images/cm_{img_path.split('/')[-1]}"
    shutil.copy(img_path, parse_location)
    cv2.imwrite(parse_location, image_resize(cv2.imread(parse_location), height=256))
    data = {
        "text": f"processed in {round(((end-start)/ 60),2)} mintues",
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
        "text": f"processed in {round(((end-start)/ 60),2)} mintues",
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
    end = time.time()
    parse_location = f"static/images/paired_{img_name}"
    shutil.copy(join("datalake", "image", img_name), parse_location)
    cv2.imwrite(parse_location, image_resize(cv2.imread(parse_location), height=256))
    data = {
		"text": f"processed in {round(((end-start)/ 60),2)} mintues",
		"image": parse_location
	}
    print(data)
    os.remove("name.txt")
    f = open("name.txt", "a")
    f.write(img_name)
    f.close()
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
    end = time.time()
    parse_location = f"static/images/unpaired_{img_name}"
    shutil.copy(join("datalake", "cloth", img_name), parse_location)
    cv2.imwrite(parse_location, image_resize(cv2.imread(parse_location), height=256))
    data = {
		"text": f"processed in {round(((end-start)/ 60),2)} mintues",
		"image": parse_location
	}
    # Return a response to the front end
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)