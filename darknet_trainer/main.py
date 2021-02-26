import sys
import asyncio
import threading
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from threading import Thread
import backdoor_controls
from fastapi_utils.tasks import repeat_every
import simplejson as json
import requests
import io
from learning_loop_node.node import Node, State
import results
import os
from typing import List
import yolo_converter

hostname = 'backend'
node = Node(hostname, uuid='c34dc41f-9b76-4aa9-8b8d-9d27e33a19e4', name='darknet trainer')


def return_true():
    return True


@node.begin_training
def begin_training(data):
    image_folder = _create_image_folder(node.status.organization, node.status.project)
    resources_ids = _extract_ressoure_ids(data)
    _download_images(node.hostname, resources_ids, image_folder)
    _update_yolo_boxes(image_folder, data)


def _extract_ressoure_ids(data) -> List[str]:
    return [(i['resource'], i['id']) for i in data['images']]


def _create_image_folder(organization: str, project: str) -> str:
    image_folder = f'/data/{organization}/{project}/images'
    os.makedirs(image_folder, exist_ok=True)
    return image_folder


def _download_images(hostname: str, resources_ids: List[tuple], image_folder: str):
    print(resources_ids, flush=True)
    for resource, image_id in resources_ids:
        url = f'http://{hostname}/api{resource}'
        response = requests.get(url)
        if response.status_code == 200:
            try:
                with open(f'/{image_folder}/{image_id}.jpg', 'wb') as f:
                    f.write(response.content)
            except IOError:
                print(f"Could not save image with id {image_id}")
        else:
            # TODO How to deal with this kind of error?
            pass


def _update_yolo_boxes(image_folder: str, data: dict):
    category_ids = [c['id']for c in data['box_categories']]

    for image in data['images']:
        image_width, image_height = image['width'], image['height']
        image_id = image['id']
        yolo_boxes = []
        for box in image['box_annotations']:
            yolo_box = yolo_converter.to_yolo(box, image_width, image_height, category_ids)
            yolo_boxes.append(yolo_box)

        with open(f'{image_folder}/{image_id}.txt', 'w') as f:
            f.write('\n'.join(yolo_boxes))


@ node.stop_training
def stop():
    # nothing to do for the darknet trainer
    pass


@ node.get_weightfile
def get_weightfile(organization: str, project: str, model_id: str) -> io.BufferedRandom:
    fake_weight_file = open('/tmp/fake_weight_file', 'wb+')
    fake_weight_file.write(b"\x00\x00\x00\x00\x00\x00\x00\x00\x01\x01\x01\x01\x01\x01")
    return fake_weight_file


@ node.on_event("startup")
@ repeat_every(seconds=5, raise_exceptions=True, wait_first=True)
async def step() -> None:
    """creating new model every 5 seconds for the demo project"""
    if node.status.model and node.status.project == 'pytest':
        await results.increment_time(node)


@ node.on_event("shutdown")
async def shutdown():

    def restart():
        asyncio.create_task(node.sio.disconnect())

    Thread(target=restart).start()


# setting up backdoor_controls
node.include_router(backdoor_controls.router, prefix="")
