import os
from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from werkzeug.utils import secure_filename
from inf_pgn import infere_parser
import shutil
import cv2
from os.path import join
import time
from flask import send_from_directory

from api_functions import gray_agnostic, detectron_densepose, detectron_poses, send_to_diffusion2
from comman_areas_refining import *
from DB_manager import insert_rows,clothes_table_columns,clothes_table_name,images_table_columns_v2,images_table_name, get_cloth_id_by_name,get_image_id_by_name
from datetime import datetime
from distutils.dir_util import copy_tree
app = Flask(__name__)


app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # for many images


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])



def save_to_DB_all_prerequesite():
    out_paths = ["image-parse-v3","openpose_img","openpose_json","agnostic-v3.2","agnostic-mask","image-densepose","cloth"]
    for folder in out_paths:
        copy_tree(f"datalake_folder/{folder}", f"/dev/MY_DB/{folder}")
        shutil.rmtree(f"datalake_folder/{folder}")
        os.mkdir(f"datalake_folder/{folder}")
    
    

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/serve_images/<path:filename>')
def serve_images(filename):
    return send_from_directory('datasets/HR_VITON_group', filename)

@app.route('/serve_images_stable/<path:filename>')
def serve_images_stable(filename):
    return send_from_directory('samples/unpair', filename)


@app.route('/server_2_call', methods=['POST'])
def upload_image():
    type = request.form['type']
    root = ""
    if type == 'SV':
        root = "/dev/MY_DB/vitons_results"
    
    uploaded_image = request.files['image']
    uploaded_file_name = request.form['path']
    uploaded_image.save(join(root,uploaded_file_name))  # Save the image to a file
    if type =='SV':
        tmp = uploaded_file_name[:-4].split("_")
        image_name, cloth_name = tmp[0], tmp[1]
        to_DB = [get_image_id_by_name(image_name), get_cloth_id_by_name(cloth_name),uploaded_file_name]
        vitons_table_columns = ["image_id","cloth_id","viton_path"]
        vitons_table_name = 'vitons'
        insert_rows(vitons_table_name,vitons_table_columns,[to_DB])

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


@app.route('/prerequiste')
def prerequiste():
    # Fetch data for button 1 (replace this with your logic)
    start = time.time()
    #step #1
    count = 5#infere_parser()
    # # step 2
    # detectron_poses()
    # #step 3
    # gray_agnostic()
    # #step 4
    # detectron_densepose()
    #step 5
    save_to_DB_all_prerequesite()

    end = time.time()
    print(count)
    data = {
        "text": f"processed {count} images in {round((end-start),2)} seconds",
    }
    print(data)
    return jsonify(data)


@app.route('/post_processing')
def post_processing():
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


@app.route('/get_all_stable_images')
def get_all__stable_images():
    # Fetch image paths from your server or database dynamically
    # This is just an example; replace it with your actual logic
    print("recieved the request")
    image_folder = 'samples/unpair'  # Change this to your actual image folder path
    image_paths = [f'/serve_images_stable/{img}' for img in os.listdir(image_folder)]
    print("recieved the request with images", image_paths)
    return jsonify({'imagePaths': image_paths})



# Define a route for handling image uploads
@app.route('/upload_person', methods=['POST'])
def upload_person():
    uploaded_files = request.files.getlist('person_images')
    print(uploaded_files)
    c = 0
    images = []
    # Process the uploaded files (you can loop through the files and handle them as needed)
    for file in uploaded_files:
        # Save or process each file
        image_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+ "_" +file.filename.split("/")[-1]
        file.save('datalake_folder/image/' + image_name)
        file.save('/dev/MY_DB/image/' + image_name)        
        images.append([image_name])

        send_to_diffusion2(join("datalake_folder", "image", image_name),"person")
        c+=1

    # TODO: to be added for the database
    insert_rows(images_table_name,images_table_columns_v2,images)
    return jsonify({'text': f'{c} Person images uploaded successfully'})
        

# Define a route for handling image uploads
@app.route('/upload_cloth', methods=['POST'])
def upload_cloth():
    uploaded_files = request.files.getlist('cloth_images')
    c= 0
    clothes = []
    # Process the uploaded files (you can loop through the files and handle them as needed)
    for file in uploaded_files:
        # Save or process each file
        image_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+ "_" +file.filename.split("/")[-1]
        file.save('datalake_folder/cloth/' + image_name)
        file.save('/dev/MY_DB/cloth/' + image_name)        
        clothes.append([image_name])

        send_to_diffusion2(join("datalake_folder", "cloth", image_name),"cloth")

        c+=1
    # TODO: to be added for the database
    insert_rows(clothes_table_name,clothes_table_columns,clothes)  
    return jsonify({'text': f'{c} cloth images uploaded successfully'})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5903)