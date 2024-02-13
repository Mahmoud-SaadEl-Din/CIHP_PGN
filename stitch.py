import os
import cv2
import numpy as np

# Path to the folders containing images
folder1_path = 'datalake_folder/openpose_img'
folder2_path = 'datalake_folder/agnostic-v3.2'

# Output folder for the stitched images
output_folder = 'datalake_folder/stitch_agnostic_pose'

# Ensure the output folder exists, create if not
os.makedirs(output_folder, exist_ok=True)

# Get a list of common image filenames in both folders
# common_filenames = set(os.listdir(folder1_path)) & set(os.listdir(folder2_path))

for filename in os.listdir(folder1_path):
    # Load images from both folders
    img1 = cv2.imread(os.path.join(folder1_path, filename))
    img2 = cv2.imread(os.path.join(folder2_path, filename.replace(".png",".jpg")))

    # Check if both images are successfully loaded
    if img1 is not None and img2 is not None:
        # Concatenate images horizontally
        concatenated_img = np.concatenate((img1, img2), axis=1)

        # Save the stitched image to the output folder
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, concatenated_img)

        print(f'Stitched and saved: {output_path}')
    else:
        print(f'Error loading images for filename: {filename}')

print('Stitching complete!')