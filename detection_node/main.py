import sys
import asyncio
import threading
from fastapi import APIRouter, FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from threading import Thread
from fastapi_utils.tasks import repeat_every
import simplejson as json
import requests
import io
from learning_loop_node.node import Node, State
from typing import List
import cv2
from glob import glob
import inferences_helper
import PIL.Image
import numpy as np
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

hostname = 'backend'
node = Node(hostname, uuid='12d7750b-4f0c-4d8d-86c6-c5ad04e19d57', name='detection node')
node.path = '/data/yolo4_tiny_3lspp_12_76844'
node.net = inferences_helper.load_network(
    f'{node.path}/training.cfg', f'{node.path}/training_final.weights')


@node.get_model_files
def get_model_files(ogranization: str, project: str, model_id: str) -> List[str]:
    return _get_model_files


def _get_model_files() -> List[str]:
    files = glob('/data/**/*')
    return files


router = APIRouter()


@router.post("/images/")
async def compute_detections(request: Request, file: List[UploadFile] = File(...)):
    inferences = []
    for image_data in file:
        try:
            image = PIL.Image.open(image_data.file)
        except:
            raise Exception(f'Uploaded file {image_data} is no image file.')
        image = np.asarray(image)
        outs = inferences_helper.get_inferences(node.net, image)
        indices, class_ids, boxes, confidences = inferences_helper.parse_inferences(outs, node.net, 608, 608)

        json_object = inferences_helper.convert_to_json(indices, class_ids, boxes, confidences)
        inferences += [json_object]

    json_compatible_item_data = jsonable_encoder(inferences)
    return JSONResponse(content=json_compatible_item_data)

# setting up backdoor_controls
node.include_router(router, prefix="")
