from api_functions import gray_agnostic_pure
from inf_pgn import infere_parser
import time
from os.path import join
import os
from infer_cloth_mask import infere_cloth_mask
from extract_warped_cloth import get_warped

def get_len(path):
    return len(os.listdir(path))

def prerequiste(id,root):
    # Fetch data for button 1 (replace this with your logic)
    start = time.time()
    #step #1
    page_index = id
    print(f"working with page {page_index}")
    # root = f"/dev/shm/LC_scraping/turkish/correct_scraping/downloads/page_{page_index}/paired"
    root = f"{root}/page_{page_index}/paired"
    
    
    in_ = join(root,"images")
    in_cloth = join(root,"cloth")
    out_cloth = join(root, "cloth-mask")
    out_ =join(root,"image-parse-v3")
    out_warped =join(root,"gt_warped")
    
    out_pose = join(root,"pose")
    out_pose_json = join(root,"openpose_json")
    out_agnostic = join(root,"agnostic")
    out_agnostic_mask = join(root,"agnostic-mask")
    out_densepose = join(root,"image-densepose")
    
    os.makedirs(out_,exist_ok=True)
    os.makedirs(out_warped,exist_ok=True)
    
    os.makedirs(out_pose,exist_ok=True)
    os.makedirs(out_pose_json,exist_ok=True)
    os.makedirs(out_agnostic,exist_ok=True)
    os.makedirs(out_agnostic_mask,exist_ok=True)
    os.makedirs(out_densepose,exist_ok=True)
    os.makedirs(out_cloth,exist_ok=True)
    
    if len(os.listdir(in_)) != len(os.listdir(out_warped)):
        print("parser section.",len(os.listdir(in_)),len(os.listdir(out_warped)))
        count = infere_parser(in_,out_)
        get_warped(out_,out_warped)
    # # step 2
    if len(os.listdir(out_agnostic)) == 0:
    #     # detectron_poses(in_,out_pose,out_pose_json)
    #     #step 3
        gray_agnostic_pure(root)
    # #step 4
    # if len(os.listdir(in_)) != len(os.listdir(out_densepose)):
    #     infer_densepose(in_,out_densepose)
    # if len(os.listdir(in_cloth)) != len(os.listdir(out_cloth)):
    #     infere_cloth_mask(in_cloth, out_cloth)
    print(f"count of images {get_len(in_)}, parsed {get_len(out_)}, pose_json {get_len(out_pose_json)}, ")
    print(f"count of images {get_len(in_)}, warped {get_len(out_warped)}, agnostic {get_len(out_agnostic)}, agnostic-mask {get_len(out_agnostic_mask)}, densepose {get_len(out_densepose)}, cloth {get_len(in_cloth)}, cloth_mask {get_len(out_cloth)}")
    
    # import shutil
    # shutil.rmtree(out_)
    # shutil.rmtree(out_pose)
    # shutil.rmtree(out_pose_json)
    
# categories = ["polo","vneck", "crew","shirts","pants","trousers","chino","jacket","vest","sweatshirt",
#                   "sweatpants","shorts","beachwear","sportswear","cardigans","jumpers","coats","pyjamas",
#                   "underwear","socks","boxer","undershirt","trainers","hats","bags"]

categories = ["pants","trousers","chino","sweatpants","shorts"]

r = "/media/HDD2/VITON/LC_WIKI_data/men_turkey"
for cat in os.listdir(r):
    root = f"/media/HDD2/VITON/LC_WIKI_data/men_turkey/{cat}/downloads"  
    print("working with Category", cat, "with root" , root)  
    for i in range(1, len(os.listdir(root))+1):
        prerequiste(i,root=root)
# error in 2(warped = 0), 6(agnostic 150/317),  