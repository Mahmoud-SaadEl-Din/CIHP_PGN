from flask import Flask, render_template, request
import os
import cv2
from os.path import join
from inf_pgn import infere_parser

app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for handling image uploads
@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Handle uploaded files here
        uploaded_files = request.files.getlist('file[]')
        # Process the uploaded files
        # Example: save files to a folder
        for file in uploaded_files:
            file.save('uploads/' + file.filename)
        infere_parser("uploads","outputs")


        return 'Files uploaded successfully'

if __name__ == '__main__':
    app.run(debug=True)