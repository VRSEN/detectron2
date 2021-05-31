import argparse
import os
from typing import Dict, List, Tuple
import torch
from torch import Tensor, nn

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, detection_utils
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.export import (
    Caffe2Tracer,
    TracingAdapter,
    add_export_config,
    dump_torchscript_IR,
    scripting_with_instances,
)
from detectron2.modeling import GeneralizedRCNN, RetinaNet, build_model
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.projects.point_rend import add_pointrend_config
from detectron2.structures import Boxes
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

def setup_cfg(args):
    cfg = get_cfg()
    # cuda context is initialized before creating dataloader, so we don't fork anymore
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg = add_export_config(cfg)
    add_pointrend_config(cfg)
    cfg.merge_from_file(args['config_file'])
    cfg.merge_from_list(args['opts'])
    cfg.freeze()
    return cfg

def get_test_loader():
    args = dict(config_file='/falldetector/detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml', export_method='tracing', format='torchscript', opts=['MODEL.WEIGHTS', 'detectron2://COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x/138363331/model_final_997cc7.pkl', 'MODEL.ROI_HEADS.SCORE_THRESH_TEST', '0.7', 'MODEL.DEVICE', 'cuda'], output='./output_tracing_float', run_eval=True, sample_image=None)

    cfg = setup_cfg(args)
    
    dataset = cfg.DATASETS.TEST[0]
    data_loader = build_detection_test_loader(cfg, dataset)
    return data_loader