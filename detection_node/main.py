import uvicorn
from fastapi import APIRouter, Request, File, UploadFile, Header
from fastapi.encoders import jsonable_encoder
from learning_loop_node.node import Node
from typing import Optional, List, Any
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
    node.model = detections_helper.setup_model(node.net, node.path)
except Exception as e:
    ic(f'Error: could not load model: {e}')


router = APIRouter()

learners = {}


@router.put("/reset")
def reset_test_learner(request: Request):
    global learners
    learners = {}


@router.post("/images")
async def compute_detections(request: Request, file: UploadFile = File(...), tags: Optional[str] = Header(None)):
    """
    Example Usage

        curl --request POST -F 'file=@example1.jpg' localhost:8004/images
    """

    try:
        np_image = np.fromfile(file.file, np.uint8)
    except:
        raise Exception(f'Uploaded file {file.filename} is no image file.')

    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    detections = get_detections(image)

    if tags:
        learn(detections, tags, image, str(file.filename))

    return JSONResponse({'box_detections': jsonable_encoder(detections)})


def get_detections(image: Any) -> List[d.Detection]:
    category_names = detections_helper.get_category_names(node.path)
    classes, confidences, boxes = detections_helper.get_inferences(node.model, image)
    net_id = detections_helper._get_model_id(node.path)
    detections = detections_helper.parse_detections(
        zip(classes, confidences, boxes), node.net, category_names, net_id)

    return detections


def learn(detections: List[d.Detection], tags: str, image: Any, filename: str) -> None:
    # TODO geht das hier async ?
    tags_list = tags.split(',')
    mac = [item for item in tags_list if item.count(':') > 2]
    mac_or_first_tag = mac[0] if mac else tags_list[0]
    active_learning_causes = check_detections_for_active_learning(detections, mac_or_first_tag)

    if any(active_learning_causes):
        helper.save_detections_and_image('/data', detections, image, filename,
                                         tags_list)  # TODO active_learning_causes as tags


def check_detections_for_active_learning(detections: List[d.Detection], mac: str) -> List[str]:
    global learners
    {learner.forget_old_detections() for (mac, learner) in learners.items()}
    if mac not in learners:
        learners[mac] = l.Learner()

    active_learning_causes = learners[mac].add_detections(detections)
    return active_learning_causes


@node.on_event("startup")
@repeat_every(seconds=30, raise_exceptions=False, wait_first=False)
def handle_detections() -> None:
    _handle_detections()


def _handle_detections() -> None:  # TODO move
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
