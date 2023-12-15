import os

import cv2
import numpy as np
from os.path import join
from typing import ClassVar, Dict

from detectron2.config import get_cfg
from detectron2.structures.instances import Instances
from detectron2.engine.defaults import DefaultPredictor

from densepose import add_densepose_config
from densepose.vis.base import CompoundVisualizer
from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
from densepose.vis.extractor import CompoundExtractor, create_extractor

from densepose.vis.densepose_results import (
    DensePoseResultsContourVisualizer,
    DensePoseResultsFineSegmentationVisualizer,
    DensePoseResultsUVisualizer,
    DensePoseResultsVVisualizer,
)

cfg = get_cfg()
add_densepose_config(cfg)

cfg.merge_from_file("densepose_rcnn_R_50_FPN_s1x.yaml")
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

cfg.MODEL.WEIGHTS = "densepose_rcnn_R_50_FPN_s1x.pkl"
predictor = DefaultPredictor(cfg)

VISUALIZERS: ClassVar[Dict[str, object]] = {
    "dp_contour": DensePoseResultsContourVisualizer,
    "dp_segm": DensePoseResultsFineSegmentationVisualizer,
    "dp_u": DensePoseResultsUVisualizer,
    "dp_v": DensePoseResultsVVisualizer,
    "bbox": ScoredBoundingBoxVisualizer,
}

vis_specs = ['dp_segm']
visualizers = []
extractors = []
for vis_spec in vis_specs:
    vis = VISUALIZERS[vis_spec]()
    visualizers.append(vis)
    extractor = create_extractor(vis)
    extractors.append(extractor)
visualizer = CompoundVisualizer(visualizers)
extractor = CompoundExtractor(extractors)

context = {
    "extractor": extractor,
    "visualizer": visualizer
}

visualizer = context["visualizer"]
extractor = context["extractor"]

captura = cv2.VideoCapture(0)


def predict(img):
    outputs = predictor(img)['instances']
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
    data = extractor(outputs)
    image_vis = visualizer.visualize(image, data)
    return image_vis

def infer_densepose(in_, out_):
    input_dir = in_
    output_dir = out_
    

    for image in os.listdir(input_dir):
        frame = cv2.imread(join(input_dir, image))
        _ = predict(frame)
        # to_be_saved = join("Pure_mask.png")
        cv2.imwrite(join(output_dir,image), predict(frame))



    captura.release()
    cv2.destroyAllWindows()