from __future__ import print_function
import argparse
from datetime import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from os.path import join
import sys
import time
import scipy.misc
import scipy.io as sio
import cv2
import argparse
from glob import glob
import time
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import logging
tf.get_logger().setLevel(logging.ERROR)

import numpy as np
from PIL import Image
from utils.image_reade_inf import *
from utils.ops import  *
from utils.utils import *
from utils.model_pgn import *

argp = argparse.ArgumentParser(description="Inference pipeline")
argp.add_argument('-i',
                  '--directory',
                  type=str, help='Path of the input dir',
                  default='./datasets/images')
argp.add_argument('-o',
                  '--output',
                  type=str, help='Path of the input dir',
                  default='./datasets/output')

args = argp.parse_args()


# Set up tf session and initialize variables. 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
init = tf.global_variables_initializer()
sess.run(init)
sess.run(tf.local_variables_initializer())

def infere_parser(in_path, out_):
    start = time.time()
    base = in_path
    image_list_inp = []
    for img in os.listdir(base):
        image_list_inp.append(join(base, img))
    print(image_list_inp)
   

    N_CLASSES = 20
    NUM_STEPS = len(image_list_inp)

    """Create the model and start the evaluation process."""
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(image_list_inp, None, False,
                             False, False, coord)
        image = reader.image
        image_rev = tf.reverse(image, tf.stack([1]))
        image_list = reader.image_list

    out_dir = out_ #"/root/diffusion_root/TestSet_LC/image-parse-v3"

    image_batch = tf.stack([image, image_rev])
    h_orig, w_orig = tf.to_float(tf.shape(image_batch)[1]), tf.to_float(tf.shape(image_batch)[2])
    scale = 0.5
    image_batch050 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, scale)), tf.to_int32(tf.multiply(w_orig, scale))]))
         
    # Create network.
    with tf.variable_scope('', reuse=False):
        net_100 = PGNModel({'data': image_batch050}, is_training=False, n_classes=N_CLASSES)
    end = time.time()
    print(f"[Not Solved]first bottle neck {round((end-start)/ 60,2)} minutes")
    
    # parsing net
    parsing_out1_100 = net_100.layers['parsing_fc']
    
    # combine resize
    parsing_out1 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out1_100, tf.shape(image_batch)[1:3,])]), axis=0)
    
    head_output, tail_output = tf.unstack(parsing_out1, num=2, axis=0)
    tail_list = tf.unstack(tail_output, num=20, axis=2)
    tail_list_rev = [None] * 20
    tail_list_rev[:14] = tail_list[:14]
    tail_list_rev[14] = tail_list[15]
    tail_list_rev[15] = tail_list[14]
    tail_list_rev[16] = tail_list[17]
    tail_list_rev[17] = tail_list[16]
    tail_list_rev[18] = tail_list[19]
    tail_list_rev[19] = tail_list[18]
    tail_output_rev = tf.stack(tail_list_rev, axis=2)
    tail_output_rev = tf.reverse(tail_output_rev, tf.stack([1]))
    raw_output_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
    raw_output_all = tf.expand_dims(raw_output_all, dim=0)
    pred_scores = tf.reduce_max(raw_output_all, axis=3)
    raw_output_all = tf.argmax(raw_output_all, axis=3)
    pred_all = tf.expand_dims(raw_output_all, dim=3) # Create 4-d tensor.

    start2 = time.time()
    # Which variables to load.
    restore_var = tf.global_variables()
    # Load weights.
    RESTORE_FROM = './checkpoint/CIHP_pgn'
    loader = tf.compat.v1.train.Saver(var_list=restore_var)
    if RESTORE_FROM is not None:
        if load(loader, sess, RESTORE_FROM):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
    # Start queue threads.
            
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    end2 = time.time()
    print(f"[Not solved]second bottle neck {round((end2-start2)/ 60,2)} minutes")
    img_id = ""
    print("before start", NUM_STEPS)
    for step in range(NUM_STEPS):
        print("Started",step)
        start = time.time()
        parsing_, scores = sess.run([pred_all, pred_scores])
        end = time.time()
        print(f"[Solved]third bottle neck {round((end-start)/ 60,2)} minutes")
        img_split = image_list[step].split('/')
        img_id = img_split[-1][:-4]

        t =parsing_[0, :, :, 0]
        with open('{}/{}_vis2.npy'.format(out_dir, img_id), 'wb') as f:
            np.save(f, t)

        
        msk = decode_labels(parsing_, num_classes=N_CLASSES)

        parsing_im = Image.fromarray(msk[0])
        parsing_im = parsing_im.convert('P')
        parsing_im.save('{}/{}_vis.png'.format(out_dir, img_id))
  
    
    coord.request_stop()
    coord.join(threads)

    return '{}/{}_vis.png'.format("./datalake/image-parse-v3", img_id)
    


if __name__ == '__main__':
    start = time.time()
    in_path, out_ = "/root/diffusion_root/CIHP_PGN/datalake/image", "/root/diffusion_root/CIHP_PGN/datalake/image-parse-v3"
    infere_parser(in_path, out_)
    end = time.time()
    print(f"time taken {round((end-start)/ 60,2)} minutes")

#################################################################
