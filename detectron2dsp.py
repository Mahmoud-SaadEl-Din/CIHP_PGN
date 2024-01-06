import cv2
import numpy as np
import torch
import detectron2
from os.path import join
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes


class Detectron2Inference:
    def __init__(self, task='detection', model=None):
        self.task = task
        self.model = model
        
        # Set up logger
        setup_logger() #The logger is responsible for printing progress and error messages to the console during the training or inference process.

        # Load configuration and model weights
        cfg = get_cfg()
        cfg.MODEL.DEVICE = 'cpu'
        if self.task == 'detection':
            if self.model is None:
                self.model = 'faster_rcnn_R_50_FPN_3x'
            cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/" + self.model + ".yaml"))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/" + self.model + ".yaml")
            
        elif self.task == 'segmentation':
            if self.model is None:
                self.model = 'mask_rcnn_R_50_FPN_3x'
            cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/" + self.model + ".yaml"))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/" + self.model + ".yaml")
            
        elif self.task == 'pose_estimation':
            if self.model is None:
                self.model = 'keypoint_rcnn_R_50_FPN_3x'
            cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/" + self.model + ".yaml"))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/" + self.model + ".yaml")
        else:
            raise ValueError('Invalid task specified.')

        # Create predictor
        self.predictor = DefaultPredictor(cfg)

        # Get metadata
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

    def infer(self, image):
        if self.task == 'detection':
            return self.detect(image)
        elif self.task == 'segmentation':
            return self.segment(image)
        elif self.task == 'pose_estimation':
            return self.pose(image)
        else:
            raise ValueError('Invalid task specified.')

    def detect(self, image):
        # Make prediction
        outputs = self.predictor(image)

        # Extract predicted boxes, classes, and scores
        boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        classes = outputs["instances"].pred_classes.cpu().numpy()
        scores = outputs["instances"].scores.cpu().numpy()

        # Visualize predictions
        v = Visualizer(image[:, :, ::-1], self.metadata, scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        output_image = v.get_image()[:, :, ::-1]

        return boxes, classes, scores, output_image

    def segment(self, image):
        # Make prediction
        outputs = self.predictor(image)
        # Extract predicted masks, classes, and scores
        masks = outputs["instances"].pred_masks.cpu().numpy()
        classes = outputs["instances"].pred_classes.cpu().numpy()
        scores = outputs["instances"].scores.cpu().numpy()

        # Visualize predictions
        v = Visualizer(image[:, :, ::-1], self.metadata, scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        output_image = v.get_image()[:, :, ::-1]

        return masks, classes, scores, output_image

    def pose(self, image):
        # Make prediction
        outputs = self.predictor(image)
        # Extract predicted keypoints, scores, and classes
        keypoints = outputs["instances"].pred_keypoints.cpu().numpy()
        scores = outputs["instances"].scores.cpu().numpy()
        classes = outputs["instances"].pred_classes.cpu().numpy()
        print("keypoints", keypoints)
        print("scores", scores)
        print("classes", classes)
        # Visualize predictions
        v = Visualizer(image[:, :, ::-1], self.metadata, scale=1)
        # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        v = v.draw_and_connect_keypoints(keypoints[0])
        output_image = v.get_image()[:, :, ::-1]

        return keypoints, scores, classes, output_image

#keypoint_rcnn_R_50_FPN_3x
#mask_rcnn_R_50_FPN_3x
#faster_rcnn_R_50_FPN_3x
import time
s = time.time()
detector = Detectron2Inference(task='pose_estimation', model='keypoint_rcnn_R_50_FPN_3x')


im_name = "t12"
image = cv2.imread(f"/root/diffusion_root/CIHP_PGN/datalake_folder/image/{im_name}.png")
out_path = "datalake_folder/openpose_json"
# Perform object detection
keypoints, classes, scores, output_image = detector.infer(image)
#output_image = cv2.resize(output_image , (1200,800))
with open(join(out_path, im_name+".npy"), 'wb') as f:
    np.save(f, keypoints[0])

print(keypoints[0])
pose_data = keypoints[0].reshape((-1, 3))[:, :2]
print(pose_data)

print(type(keypoints[0]))
for i, pnt in enumerate(keypoints[0]):
    image = cv2.circle(image, (int(pnt[0]),int(pnt[1])), radius=1, color=(0, 0, 255), thickness=-1)
    image = cv2.putText(image, f'{i}', (int(pnt[0])-5,int(pnt[1])-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
    cv2.imwrite("key_points_detectron2.png", image)

print(time.time() -s , "seconds")
# Display output image
cv2.imwrite("Output.png", output_image)