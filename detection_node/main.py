from fastapi import APIRouter, Request, File, UploadFile
from learning_loop_node.node import Node
from typing import List
import cv2
from glob import glob
import detections_helper
import PIL.Image
import numpy as np
from fastapi.responses import JSONResponse
import helper
from icecream import ic

node = Node(uuid='12d7750b-4f0c-4d8d-86c6-c5ad04e19d57', name='detection node')
node.path = '/model'
try:
    node.net = detections_helper.load_network(
        helper.find_cfg_file(node.path), helper.find_weight_file(node.path))
except Exception as e:
    ic(f'Error: could not load model: {e}')


router = APIRouter()


@router.post("/images")
async def compute_detections(request: Request, file: UploadFile = File(...)):
    """
    Example Usage

        curl --request POST -F 'file=@example1.jpg' localhost:8004/detect
    """

    try:
        np_image = np.fromfile(file.file, np.uint8)
    except:
        raise Exception(f'Uploaded file {file.filename} is no image file.')

    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    net_input_image_width, net_input_image_height = detections_helper.get_network_input_image_size(node.path)
    category_names = detections_helper.get_category_names(node.path)
    classes, confidences, boxes = detections_helper.get_inferences(
        node.net, image, net_input_image_width, net_input_image_height)
    net_id = detections_helper._get_model_id(node.path)
    detections = detections_helper.parse_detections(
        zip(classes, confidences, boxes), node.net, category_names, image.shape[1], image.shape[0], net_id)

    return JSONResponse({'box_detections': detections})

node.include_router(router, prefix="")
