import cv2
from typing import List, Any
from typing import Union as UNION
from icecream import ic
import helper
import os
import detection as d
import subprocess
from ctypes import *
import c_classes


lib = CDLL("/tkDNN/build/libdarknetRT.so", RTLD_GLOBAL)


load_network = lib.load_network
load_network.argtypes = [c_char_p, c_int, c_int]
load_network.restype = c_void_p

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [c_classes.IMAGE, c_char_p]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = c_classes.IMAGE

do_inference = lib.do_inference
do_inference.argtypes = [c_void_p, c_classes.IMAGE]

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_float, POINTER(c_int)]
get_network_boxes.restype = POINTER(c_classes.DETECTION)


def get_number_of_classes(model_path: str) -> int:
    with open(f'{model_path}/names.txt', 'r') as f:
        names = f.read().rstrip('\n').split('\n')
    return len(names)


def export_weights(cfg_file_path: str, weightfile_path: str):
    cmd = f'cd /tkDNN/darknet; ./darknet export {cfg_file_path} {weightfile_path} layers'
    p = subprocess.Popen(cmd, shell=True)
    _, err = p.communicate()
    if p.returncode != 0:
        raise Exception(f'Failed to export weights: {err}')
    return p.returncode


def create_rt_file():
    cmd = f'cd /tkDNN/build; ./test_yolo4tiny'
    p = subprocess.Popen(cmd, shell=True)
    _, err = p.communicate()
    if p.returncode != 0:
        raise Exception(f'Failed to create .rt file: {err}')
    return p.returncode


def load_network_file(rt_file_path: str, model_path: str) -> None:
    net = load_network(rt_file_path.encode("ascii"), get_number_of_classes(model_path), 1)
    return net


def create_darknet_image(image: Any) -> None:
    try:
        height, width, channels = image.shape
        darknet_image = make_image(width, height, channels)
        frame_data = image.ctypes.data_as(c_char_p)
        copy_image_from_bytes(darknet_image, frame_data)
    except Exception as e:
        print(e)

    return darknet_image


def get_detections(image: Any, net: bytes, model_path: str) -> List[d.Detection]:
    darknet_image = create_darknet_image(image)
    net_id = get_model_id(model_path)
    detections = detect_image(net, darknet_image, net_id)
    parsed_detections = parse_detections(detections)
    return parsed_detections


def detect_image(net, darknet_image, net_id, thresh=.2):
    num = c_int(0)

    pnum = pointer(num)
    do_inference(net, darknet_image)
    dets = get_network_boxes(net, thresh, pnum)
    detections = []
    for i in range(pnum[0]):
        bbox = dets[i].bbox
        detections.append((dets[i].name.decode("ascii"), dets[i].prob, bbox.x, bbox.y, bbox.w, bbox.h, net_id))

    return detections


def parse_detections(detections: List[UNION[int, str]]) -> List[d.Detection]:
    parsed_detections = []
    for detection in detections:
        category_name = detection[0]
        confidence = round(detection[1], 3) * 100
        left = int(detection[2])
        top = int(detection[3])
        width = int(detection[4])
        height = int(detection[5])
        net_id = detection[6]
        detection = d.Detection(category_name, left, top, width, height, net_id, confidence)
        parsed_detections.append(detection)

    return parsed_detections


def get_model_id(model_path: str) -> str:
    weightfile = helper.find_weight_file(model_path)
    return os.path.basename(weightfile).split('.')[0]
