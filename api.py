import os
from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from werkzeug.utils import secure_filename
from inf_pgn import infere_parser
import shutil
import cv2
from os.path import join
import time
from flask import send_from_directory

from api_functions import gray_agnostic, detectron_densepose, detectron_poses, send_to_diffusion2,ask_server2_to_diffuse,poses_classification_with_sm
from comman_areas_refining import *
from DB_manager import insert_rows,clothes_table_columns,clothes_table_name,images_table_columns_v2,images_table_name, get_cloth_id_by_name,get_image_id_by_name, get_cloth_name_by_id, get_image_name_by_id
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
    return send_from_directory('/dev/MY_DB/viton_results', filename)

@app.route('/serve_images_stable/<path:filename>')
def serve_images_stable(filename):
    return send_from_directory('samples/unpair', filename)


@app.route('/server_2_call', methods=['POST'])
def upload_image():
    type = request.form['type']
    root = ""
    if type == 'SV':
        root = "/dev/MY_DB/viton_results"
    
    uploaded_image = request.files['image']
    uploaded_file_name = request.form['path']
    uploaded_image.save(join(root,uploaded_file_name))  # Save the image to a file
    if type =='SV':
        tmp = uploaded_file_name[:-4].split("**")
        image_name, cloth_name = tmp[0]+".png", tmp[1]+".png"
        take_face_from_original_to_diffusion(uploaded_file_name,image_name)
        print(image_name, cloth_name)
        to_DB = [get_image_id_by_name(image_name), get_cloth_id_by_name(cloth_name),uploaded_file_name]
        print("check here",to_DB)
        vitons_table_columns = ["image_id","cloth_id","viton_path"]
        vitons_table_name = 'vitons'
        insert_rows(vitons_table_name,vitons_table_columns,[to_DB])

    # Return a response if needed (optional)
    return 'Image received and processed on GPU server'

# Define a route for the home page
@app.route('/')
def index():
    return render_template('index_bulk.html')


# Define a route for the home page
@app.route('/bulk')
def index_bulk():
    return render_template('index_bulk.html')



@app.route('/diffuse/<int:person_id>/<int:cloth_id>')
def your_route_handler(person_id, cloth_id):
    # Your route handling logic here
    p_name, c_name = get_image_name_by_id(person_id), get_cloth_name_by_id(cloth_id)
    if p_name != None:
        send_to_diffusion2(join("/dev/MY_DB", "image", p_name),"person")
        #send data to server 2
        gray_agnostic(by_name=p_name,send=True)
        #step 4
        detectron_densepose(by_name=p_name,send=True)
    if c_name !=None:
        # send cloth to server 2
        send_to_diffusion2(join("/dev/MY_DB", "cloth", c_name),"cloth")
    
    #send request to server 2 to work on them
    ask_server2_to_diffuse()

    data = {
        "text": f"Working with IDs",
    }
    print(data)
    return jsonify(data)


@app.route('/prerequiste/<int:person_id>')
def prerequiste(person_id):
    # Fetch data for button 1 (replace this with your logic)
    p_name = get_image_name_by_id(person_id)
    clear_all()
    shutil.copy(join("/dev/MY_DB", "image", p_name), join("datalake_folder","image",p_name))
    start = time.time()
    #step #1
    count = infere_parser()
    # # step 2
    detectron_poses()
    #step 3
    gray_agnostic()
    #step 4
    detectron_densepose()
    #step 5
    save_to_DB_all_prerequesite()
    clear_all()

    end = time.time()
    print(count)
    data = {
        "text": f"processed {count} images in {round((end-start),2)} seconds",
    }
    print(data)
    return jsonify(data)


@app.route('/get_all_stable_images')
def get_all__stable_images():
    # Fetch image paths from your server or database dynamically
    # This is just an example; replace it with your actual logic
    print("recieved the request")
    image_folder = '/dev/MY_DB/viton_results'  # Change this to your actual image folder path
    image_paths = [f'/dev/MY_DB/viton_results/{img}' for img in os.listdir(image_folder)]
    print("recieved the request with images", image_paths)
    return jsonify({'imagePaths': image_paths})


# Define a route for handling image uploads
@app.route('/classify_pose', methods=['POST'])
def classify_pose():
    uploaded_files = request.files.getlist('person_images')
    print(uploaded_files)
    images = []
    clear_cache_poses()
    # Process the uploaded files (you can loop through the files and handle them as needed)
    for file in uploaded_files:
        # Save or process each file
        image_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+ "_" +file.filename.split("/")[-1]
        png_name = ''.join(image_name.split(".")[:-1]) + ".png"

        p1 = join('temp_poses/imgs', png_name)
        file.save(p1)
        img = cv2.imread(p1)
        # img = cv2.resize(img,(384,512))
        cv2.imwrite(p1,img)
    
    detectron_poses('temp_poses/imgs','temp_poses/poses_imgs','temp_poses/poses_json')
    
    names, prob, class_ = poses_classification_with_sm()
    clear_cache_poses()
    return jsonify({'text': f'{names},{prob}, {class_} poses'})
    

# Define a route for handling image uploads
@app.route('/upload_person', methods=['POST'])
def upload_person():
    uploaded_files = request.files.getlist('person_images')
    print(uploaded_files)
    images = []
    # Process the uploaded files (you can loop through the files and handle them as needed)
    for file in uploaded_files:
        # Save or process each file
        image_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+ "_" +file.filename.split("/")[-1]
        png_name = ''.join(image_name.split(".")[:-1]) + ".png"

        p1 = join('datalake_folder/image/', png_name)
        p2 = join('/dev/MY_DB/image/', png_name)
        file.save(p1)
        img = cv2.imread(p1)
        img = cv2.resize(img,(384,512))
        cv2.imwrite(p1,img)

        shutil.copy(p1,p2)        
        images.append([png_name])

    # TODO: to be added for the database
    ids = insert_rows(images_table_name,images_table_columns_v2,images)
    return jsonify({'text': f'{ids} Person images uploaded successfully'})
        

# Define a route for handling image uploads
@app.route('/upload_cloth', methods=['POST'])
def upload_cloth():
    uploaded_files = request.files.getlist('cloth_images')
    clothes = []
    # Process the uploaded files (you can loop through the files and handle them as needed)
    for file in uploaded_files:
        # Save or process each file
        image_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+ "_" +file.filename.split("/")[-1]
        png_name = ''.join(image_name.split(".")[:-1]) + ".png"

        p1 = join('datalake_folder/cloth/', png_name)
        p2 = join('/dev/MY_DB/cloth/', png_name)
        file.save(p1)
        img = cv2.imread(p1)
        img = cv2.resize(img,(384,512))
        cv2.imwrite(p1,img)

        shutil.copy(p1,p2)        
        clothes.append([png_name])

    # TODO: to be added for the database
    ids = insert_rows(clothes_table_name,clothes_table_columns,clothes)  
    return jsonify({'text': f'{ids} cloth images uploaded successfully'})



def clear_all():
    main_folder = "datalake_folder"
    
    l = ['image', 'image-parse-v3','image-parse-agnostic-v3.2', 'agnostic-v3.2', 'openpose_img', 'openpose_json','agnostic-v3.2','image-densepose']
    for name in l:
        shutil.rmtree(join(main_folder, name))
        os.mkdir(join(main_folder, name))
    l = ["cloth", "cloth-mask"]
    for name in l:     
        shutil.rmtree(join(main_folder, name))
        os.mkdir(join(main_folder, name))

def clear_cache_poses():
    main_folder = "temp_poses"
    
    l = ['imgs', 'poses_imgs','poses_json']
    for name in l:
        shutil.rmtree(join(main_folder, name))
        os.mkdir(join(main_folder, name))

if __name__ == '__main__':
    clear_all()
    app.run(debug=True, host="0.0.0.0", port=5903)