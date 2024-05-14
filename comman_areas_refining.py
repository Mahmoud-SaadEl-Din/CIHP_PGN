import os
import cv2
from os.path import join
import numpy as np

def conditional_merge(im_rgb_diffused,im_rgb_originnal,im_parse=None):
    print(im_parse.shape)
    print(im_rgb_originnal.shape, im_rgb_diffused.shape)
    wanted_ids = [1,2,3,4,13]
    for id in wanted_ids:
        im_rgb_diffused[im_parse==id] = im_rgb_originnal[im_parse==id]

    

    return im_rgb_diffused



def take_face_from_original_to_diffusion(generated_image_name, original_img_name):
    
    ori_img = cv2.imread(join("images_DB/image", original_img_name))
    ori_img = cv2.resize(ori_img,(384,512))
    diffused_img = cv2.imread(join("images_DB/viton_results",generated_image_name))
    with open(join("images_DB/image-parse-v3", original_img_name.replace(".png","_vis2.npy")), 'rb') as f:
        im_parse_np = np.load(f)
    cv2.imwrite(join("images_DB/viton_results",generated_image_name), conditional_merge(diffused_img,ori_img,im_parse_np))
    print(f'Stitched and saved: {generated_image_name}')

    



def compare_between_dense_and_parse(r2):
    img_p ="/root/diffusion_root/CIHP_PGN/datalake_folder/image" 
    img_n ="/root/diffusion_root/CIHP_PGN/datalake_folder/image_noise" 

    r1 = "/root/diffusion_root/CIHP_PGN/datalake_folder/image-densepose"
    r2 = "/root/diffusion_root/CIHP_PGN/datalake_folder/image-parse-v3"

    o1= "/root/diffusion_root/CIHP_PGN/datalake_folder/bw/densepose_bw"
    o2= "/root/diffusion_root/CIHP_PGN/datalake_folder/bw/parse_bw"
    o3 = "/root/diffusion_root/CIHP_PGN/datalake_folder/image_filtered"
    for image in os.listdir(r2):
        if "npy" not in image:
            print(image)
            im_gray = cv2.imread(join(r2,image), cv2.IMREAD_GRAYSCALE)
            im_rgb = cv2.imread(join(img_p,image.replace("_vis","").replace(".png",".jpg")))

            bw = cv2.threshold(im_gray, 7, 255, cv2.THRESH_BINARY)[1]
            # cv2.imwrite(join(o2,image.split(".")[0].split("_")[0]+"bw.png"), bw)
            segmented_image = cv2.bitwise_and(im_rgb, im_rgb, mask=bw)
            cv2.imwrite(join(o2,image.split(".")[0].split("_")[0]+"filter.png"), segmented_image)



def remove_gray_area(parsed_original_imgs,diffused_imgs,original_imgs_path,out_path):
    images_dict = {image.split(".")[0]:image for image in os.listdir(original_imgs_path)}
    for image in os.listdir(diffused_imgs):
            print(image)
            # im_gray = cv2.imread(join(parsed_original_imgs,image.split("_")[0]+"_vis.png"), cv2.IMREAD_GRAYSCALE)
            im_rgb = cv2.imread(join(diffused_imgs,image))
            # im_rgb_original = cv2.imread(join(original_imgs_path, images_dict[image.split("_")[0]]))


            # bw = cv2.threshold(im_gray, 7, 255, cv2.THRESH_BINARY)[1]
        
            # segmented_image = cv2.bitwise_and(im_rgb, im_rgb, mask=bw)
            # segmented_image[bw == 0] = im_rgb_original[bw == 0]
            # segmented_image[np.all(segmented_image == [128, 128, 128], axis=-1)] = [255,255,255]

            cv2.imwrite(join(out_path,image), im_rgb)



