import cv2
from typing import List, Tuple, Any
from icecream import ic
import numpy as np
import helper
import os
import detection as d


def get_category_names(model_path: str) -> List[str]:
    with open(f'{model_path}/names.txt', 'r') as f:
        names = f.read().rstrip('\n').split('\n')
    return names


def load_network(cfg_file_path: str, weightfile_path: str) -> cv2.dnn_Net:
    net = cv2.dnn.readNetFromDarknet(cfg_file_path, weightfile_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return net


def get_inferences(net: cv2.dnn_Net, image: Any, net_input_image_width, net_input_image_height, swapRB=False) -> List[Any]:
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(net_input_image_width, net_input_image_height), scale=1/255, swapRB=swapRB)
    classes, confidences, boxes = model.detect(image, confThreshold=0.2, nmsThreshold=1.0)
    return classes, confidences, boxes


def parse_detections(outs: List[int], net: cv2.dnn_Net, category_names: List[str], image_width: int, image_height: int, net_id: str) -> List[d.Detection]:
    detections = []
    for (class_id, confidence, box) in outs:
        category = category_names[int(class_id)]
        left = int(box[0])
        top = int(box[1])
        width = int(box[2])
        height = int(box[3])
        confidence = round(float(confidence), 3)
        detection = d.Detection(category, left, top, width, height, net_id, confidence).__dict__
        detections.append(detection)

    return detections


def _read_image(image_path: str) -> List[int]:
    return cv2.imread(image_path)


def _get_out_names(net: cv2.dnn_Net):
    return net.getUnconnectedOutLayersNames()


def _get_last_layer_type(net: cv2.dnn_Net):
    layerNames = net.getLayerNames()
    lastLayerId = net.getLayerId(layerNames[-1])
    lastLayer = net.getLayer(lastLayerId)
    return lastLayer.type


def _get_model_id(model_path: str) -> str:
    weightfile = helper.find_weight_file(model_path)
    return os.path.basename(weightfile).split('.')[0]


def get_network_input_image_size(model_path: str) -> Tuple[int, int]:
    with open(f'{model_path}/training.cfg', 'r') as f:
        content = f.readlines()

    for line in content:
        if line.startswith('width'):
            width = line.split('=')[-1].strip()
        if line.startswith('height'):
            height = line.split('=')[-1].strip()

    if not width or not height:
        raise Exception("width or height are missing in cfg file.")

    return int(width), int(height)
