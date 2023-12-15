# Set sample
import numpy as np
import pandas as pd
import os
from os.path import join
import random

def prepare_samples(root):
    image = os.listdir(join(root,"image"))
    clothes = os.listdir(join(root,"cloth"))
    random.shuffle(clothes)

    df = pd.DataFrame({"image": image, "clothes": clothes})
    df.to_csv(join(root,"pairs.txt"), index=False, header=False, sep=" ")

