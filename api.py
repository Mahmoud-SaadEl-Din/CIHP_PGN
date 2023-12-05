import os
from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from werkzeug.utils import secure_filename

import cv2
from os.path import join
from inf_pgn import infere_parser
import time
app = Flask(__name__)

app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # for many images


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


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
    img_path = infere_parser("/root/diffusion_root/CIHP_PGN/datalake/image","/root/diffusion_root/CIHP_PGN/datalake/image-parse-v3")
    end = time.time()
    data = {
        "text": f"processed in {(end-start)/ 60} mintues",
        "image": img_path
    }
    return jsonify(data)

@app.route('/route_for_button_2')
def route_for_button_2():
    # Fetch data for button 2 (replace this with your logic)
    data = {
        "text": "Text for Button 2",
        "image": "path_to_image/image2.jpg"
    }
    return jsonify(data)

@app.route('/route_for_button_3')
def route_for_button_3():
    # Fetch data for button 3 (replace this with your logic)
    data = {
        "text": "Text for Button 3",
        "image": "path_to_image/image3.jpg"
    }
    return jsonify(data)

@app.route('/route_for_button_4')
def route_for_button_4():
    # Fetch data for button 4 (replace this with your logic)
    data = {
        "text": "Text for Button 4",
        "image": "path_to_image/image4.jpg"
    }
    return jsonify(data)

@app.route('/route_for_button_5')
def route_for_button_5():
    # Fetch data for button 5 (replace this with your logic)
    data = {
        "text": "Text for Button 5",
        "image": "path_to_image/image5.jpg"
    }
    return jsonify(data)

@app.route('/route_for_button_6')
def route_for_button_6():
    # Fetch data for button 6 (replace this with your logic)
    data = {
        "text": "Text for Button 6",
        "image": "path_to_image/image6.jpg"
    }
    return jsonify(data)

@app.route('/route_for_button_7')
def route_for_button_7():
    # Fetch data for button 7 (replace this with your logic)
    data = {
        "text": "Text for Button 7",
        "image": "path_to_image/image7.jpg"
    }
    return jsonify(data)

# Define a route for handling image uploads
@app.route('/upload_person', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Handle uploaded files here
        uploaded_files = request.files.getlist('file[]')
        # Process the uploaded files
        # Example: save files to a folder
        for file in uploaded_files:
            file.save('uploads/image' + file.filename)
        return 'Files uploaded successfully'

# Define a route for handling image uploads
@app.route('/upload_cloth', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Handle uploaded files here
        uploaded_files = request.files.getlist('file[]')
        # Process the uploaded files
        # Example: save files to a folder
        for file in uploaded_files:
            file.save('datalake/cloth' + file.filename)
        return 'Files uploaded successfully'


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)