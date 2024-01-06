import cv2, os
import numpy as np
from os.path import join
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog



# Set up logger
setup_logger() #The logger is responsible for printing progress and error messages to the console during the training or inference process.
# Load configuration and model weights
cfg = get_cfg()
cfg.MODEL.DEVICE = "cpu"
model = 'keypoint_rcnn_R_50_FPN_3x'
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/" + model + ".yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/" + model + ".yaml")

# Create predictor
predictor = DefaultPredictor(cfg)

# Get metadata
metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

def pose_image(image_path):
    # Make prediction
    image = cv2.imread(image_path)
    outputs = predictor(image)
    # Extract predicted keypoints, scores, and classes
    keypoints = outputs["instances"].pred_keypoints.cpu().numpy()
    # Visualize predictions
    v = Visualizer(image[:, :, ::-1], metadata, scale=1)
    # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    v = v.draw_and_connect_keypoints(keypoints[0])
    output_image = v.get_image()[:, :, ::-1]

    return keypoints, output_image


def pose_dir(input_dir, out_dir, out_dir_json):
    
    for im_name in os.listdir(input_dir):
        name = im_name.split(".")[0]
        # Make prediction
        image = cv2.imread(join(input_dir,im_name))
        outputs = predictor(image)
        # Extract predicted keypoints, scores, and classes
        keypoints = outputs["instances"].pred_keypoints.cpu().numpy()
        # Visualize predictions
        v = Visualizer(image[:, :, ::-1], metadata, scale=1)
        # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        v = v.draw_and_connect_keypoints(keypoints[0])
        output_image = v.get_image()[:, :, ::-1]
        cv2.imwrite(join(out_dir,name+".png"), output_image)

        with open(join(out_dir_json, name+".npy"), 'wb') as f:
            np.save(f, keypoints[0])
    


def draw():
    for i, pnt in enumerate(keypoints[0]):
        image = cv2.circle(image, (int(pnt[0]),int(pnt[1])), radius=1, color=(0, 0, 255), thickness=-1)
        image = cv2.putText(image, f'{i}', (int(pnt[0])-5,int(pnt[1])-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        cv2.imwrite("key_points_detectron2.png", image)
