from fastapi import APIRouter, Request, File, UploadFile, Form
from fastapi.param_functions import Form
from learning_loop_node.node import Node
from typing import Optional
import cv2
from glob import glob
import detections_helper
import PIL.Image
import numpy as np
from fastapi.responses import JSONResponse
import helper
from icecream import ic
from fastapi_utils.tasks import repeat_every
from active_learner import learner as l
import json
from active_learner import detection as d
import requests

node = Node(uuid='12d7750b-4f0c-4d8d-86c6-c5ad04e19d57', name='detection node')
node.path = '/model'
try:
    node.net = detections_helper.load_network(
        helper.find_cfg_file(node.path), helper.find_weight_file(node.path))
except Exception as e:
    ic(f'Error: could not load model: {e}')


router = APIRouter()

learners = {}


@router.post("/images")
async def compute_detections(request: Request, file: UploadFile = File(...), mac: Optional[str] = Form(None)):
    """
    Example Usage

        curl --request POST -F 'file=@example1.jpg' localhost:8004/images
    """

    try:
        np_image = np.fromfile(file.file, np.uint8)
    except:
        raise Exception(f'Uploaded file {file.filename} is no image file.')

    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    net_input_image_width, net_input_image_height = detections_helper.get_network_input_image_size(node.path)
    category_names = detections_helper.get_category_names(node.path)
    classes, confidences, boxes = detections_helper.get_inferences(
        node.net, image, net_input_image_width, net_input_image_height, swapRB=True)
    net_id = detections_helper._get_model_id(node.path)
    detections = detections_helper.parse_detections(
        zip(classes, confidences, boxes), node.net, category_names, image.shape[1], image.shape[0], net_id)

    if mac and detections:
        helper.save_detections_and_image(detections, image, mac)

    return JSONResponse({'box_detections': detections})


@node.on_event("startup")
@repeat_every(seconds=30, raise_exceptions=False, wait_first=False)
async def handle_detections() -> None:
    files_for_active_learning = glob('/data/*', recursive=True)

    global learners
    {learner.forget_old_detections() for (mac, learner) in learners.items()}

    if files_for_active_learning:
        macs_dict = helper.extract_macs_and_filenames(files_for_active_learning)
        for mac, filenames in macs_dict.items():
            if mac not in learners:
                learners[mac] = l.Learner()

            for filename in filenames:
                with open(f'/data/{filename}.json') as f:
                    detections = json.load(f)

                active_learning_causes = learners[mac].add_detections(
                    [d.ActiveLearnerDetection(detection['category_name'], detection['x'], detection['y'], detection['width'], detection['height'], detection['model_name'], detection['confidence']) for detection in detections['box_detections']])

                if any(active_learning_causes):
                    data = [('file', open(f'/data/{filename}.json', 'r')),
                            ('file', open(f'/data/{filename}.jpg', 'rb'))]
                    requests.post('http://backend/api/zauberzeug/projects/demo/images', files=data)


node.include_router(router, prefix="")
