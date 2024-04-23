import os,glob
from os.path import join
import numpy as np
from PIL import Image
from tqdm import tqdm

def get_warped(in_,out_):
    for cls_arr in tqdm(glob.glob(f"{in_}/*.npy"), desc="Warped GT..."):
        with open(cls_arr, 'rb') as f:
            im_parse_np = np.load(f)
             # Create binary image
            binary_image = (im_parse_np == 5).astype(np.uint8) * 255
            
            # Convert to PIL Image
            binary_image_pil = Image.fromarray(binary_image)
            
            # Save or process the binary image
            # For example, save to output directory
            binary_image_pil.save(out_ + '/' + cls_arr.split('/')[-1].replace('_vis2.npy', '.png'))
