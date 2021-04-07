import uvicorn
from fastapi import APIRouter, Request, File, UploadFile, Header
from learning_loop_node.node import Node
from typing import Optional, List
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
from detection import Detection
import os

node = Node(uuid='12d7750b-4f0c-4d8d-86c6-c5ad04e19d57', name='detection node')
node.path = '/model'
try:
    node.net = detections_helper.load_network(
        helper.find_cfg_file(node.path), helper.find_weight_file(node.path))
except Exception as e:
    ic(f'Error: could not load model: {e}')


router = APIRouter()

learners = {}


@router.put("/reset")
def reset_test_learner(request: Request):
    global learners
    learners = {}


@router.post("/images")
async def compute_detections(request: Request, file: UploadFile = File(...), mac: Optional[str] = Header(None)):
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
        zip(classes, confidences, boxes), node.net, category_names, net_id)

    if mac and detections:
        active_learning_causes = check_detections_for_active_learning(detections, mac)

        if any(active_learning_causes):
            helper.save_detections_and_image('/data', detections, image, str(file.filename).rsplit('.', 1)[0], mac)

    return JSONResponse({'box_detections': detections})


def check_detections_for_active_learning(detections: dict, mac: str) -> List[str]:
    global learners
    {learner.forget_old_detections() for (mac, learner) in learners.items()}
    if mac not in learners:
        learners[mac] = l.Learner()

    active_learning_causes = learners[mac].add_detections(
        [d.ActiveLearnerDetection(Detection.from_dict(detection)) for detection in detections])

    return active_learning_causes


@node.on_event("startup")
@repeat_every(seconds=30, raise_exceptions=False, wait_first=False)
def handle_detections() -> None:
    _handle_detections()


def _handle_detections() -> None:
    files_for_active_learning = glob('/data/*', recursive=True)
    file_names = helper.get_file_paths(files_for_active_learning)
    for filename in file_names:
        data = [('file', open(f'{filename}.json', 'r')),
                ('file', open(f'{filename}.jpg', 'rb'))]
        request = requests.post(f'{node.url}/api/{node.organization}/projects/{node.project}/images', files=data)
        if request.status_code == 200:
            os.remove(f'{filename}.json')
            os.remove(f'{filename}.jpg')


node.include_router(router, prefix="")

if __name__ == "__main__":
    uvicorn.run("main:node", host="0.0.0.0", port=80, lifespan='on', reload=True)
