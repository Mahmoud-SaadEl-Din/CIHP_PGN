# Set sample
import numpy as np
import pandas as pd
import os
from os.path import join
import random

def prepare_samples(root):
    final_images = []
    final_clothes = []

    
    images = os.listdir(join(root,"image"))
    clothes = os.listdir(join(root,"cloth"))
    # random.shuffle(clothes)
    for image in images:
        for cloth in clothes:
            final_images.append(image)
            final_clothes.append(cloth)


    df = pd.DataFrame({"image": final_images, "clothes": final_clothes})
    df.to_csv(join(root,"pairs.txt"), index=False, header=False, sep=" ")

