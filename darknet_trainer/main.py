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
import helper
import yolo_helper
from uuid import uuid4

hostname = 'backend'
node = Node(hostname, uuid='c34dc41f-9b76-4aa9-8b8d-9d27e33a19e4', name='darknet trainer')


def return_true():
    return True


@node.begin_training
def begin_training(data: dict) -> None:
    project_folder = _create_project_folder(node.status.organization, node.status.project)
    image_folder = _create_image_folder(project_folder)
    image_resources = _extract_image_ressoures(data)
    image_ids = _extract_image_ids(data)
    _download_images(node.hostname, zip(image_resources, image_ids), image_folder)

    trainings_folder = _create_trainings_folder(project_folder, str(uuid4()))

    image_folder_for_training = yolo_helper.create_image_links(trainings_folder, image_folder, image_ids)

    yolo_helper.update_yolo_boxes(image_folder_for_training, data)

    box_categories = helper.get_box_category_ids(data)
    yolo_helper.create_names_file(trainings_folder, box_categories)
    yolo_helper.create_data_file(trainings_folder, len(box_categories))


def _create_project_folder(organization: str, project: str) -> str:
    project_folder = f'../data/{organization}/{project}'
    os.makedirs(project_folder, exist_ok=True)
    return project_folder


def _extract_image_ressoures(data: dict) -> List[tuple]:
    return [i['resource'] for i in data['images']]


def _extract_image_ids(data: dict) -> List[str]:
    return [i['id'] for i in data['images']]


def _create_image_folder(project_folder: str) -> str:
    image_folder = f'{project_folder}/images'
    os.makedirs(image_folder, exist_ok=True)
    return image_folder


def _download_images(hostname: str, image_ressources_and_ids: List[tuple], image_folder: str) -> None:
    for resource, image_id in image_ressources_and_ids:
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


def _create_trainings_folder(project_folder: str, trainings_id: str) -> str:
    trainings_folder = f'{project_folder}/trainings/{trainings_id}'
    os.makedirs(trainings_folder, exist_ok=True)
    return trainings_folder


@ node.stop_training
def stop() -> None:
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
