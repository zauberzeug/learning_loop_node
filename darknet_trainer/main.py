import asyncio
from threading import Thread
import backdoor_controls
from fastapi_utils.tasks import repeat_every
import requests
import io
from learning_loop_node.node import Node
import os
from typing import List
import helper
import yolo_helper
import yolo_cfg_helper
from uuid import uuid4
import shutil
from io import BytesIO
import zipfile
import os
from glob import glob
import subprocess
from icecream import ic
import psutil
from status import State

hostname = 'backend'
node = Node(hostname, uuid='c34dc41f-9b76-4aa9-8b8d-9d27e33a19e4',
            name='darknet trainer')


@node.begin_training
def begin_training(data: dict) -> None:
    training_uuid = str(uuid4())
    _prepare_training(node, data, training_uuid)
    _start_training(training_uuid)


def _prepare_training(node: Node, data: dict, training_uuid: str) -> None:
    project_folder = _create_project_folder(
        node.status.organization, node.status.project)
    image_folder = _create_image_folder(project_folder)
    image_resources = _extract_image_ressoures(data)
    image_ids = _extract_image_ids(data)
    _download_images(node.hostname, zip(
        image_resources, image_ids), image_folder)

    training_folder = _create_training_folder(project_folder, training_uuid)

    image_folder_for_training = yolo_helper.create_image_links(
        training_folder, image_folder, image_ids)

    yolo_helper.update_yolo_boxes(image_folder_for_training, data)

    box_category_names = helper._get_box_category_names(data)
    box_category_count = len(box_category_names)
    yolo_helper.create_names_file(training_folder, box_category_names)
    yolo_helper.create_data_file(training_folder, box_category_count)
    yolo_helper.create_train_and_test_file(
        training_folder, image_folder_for_training, data['images'])
    yolo_helper.create_backup_dir(training_folder)

    _download_model(training_folder, node.status.organization,
                    node.status.project, node.status.model['id'], node.hostname)
    yolo_cfg_helper.replace_classes_and_filters(
        box_category_count, training_folder)
    yolo_cfg_helper.update_anchors(training_folder)


def _create_project_folder(organization: str, project: str) -> str:
    project_folder = f'/data/{organization}/{project}'
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


def _create_training_folder(project_folder: str, trainings_id: str) -> str:
    training_folder = f'{project_folder}/trainings/{trainings_id}'
    os.makedirs(training_folder, exist_ok=True)
    return training_folder


def _download_model(training_folder: str, organization: str, project: str, model_id: str, hostname: str):
    # download model
    download_response = requests.get(
        f'http://{hostname}/api/{organization}/projects/{project}/models/{model_id}/file')
    assert download_response.status_code == 200
    provided_filename = download_response.headers.get(
        "Content-Disposition").split("filename=")[1].strip('"')

    # unzip and place downloaded model
    target_path = f'/tmp/{os.path.splitext(provided_filename)[0]}'
    shutil.rmtree(target_path, ignore_errors=True)
    filebytes = BytesIO(download_response.content)
    with zipfile.ZipFile(filebytes, 'r') as zip:
        zip.extractall(target_path)

    files = glob(f'{target_path}/**/*', recursive=True)
    for file in files:
        shutil.move(file, training_folder)


@ node.stop_training
def stop() -> None:
    # nothing to do for the darknet trainer
    pass


def _start_training(training_id: str) -> None:
    training_path = get_training_path_by_id(training_id)
    os.chdir(training_path)
    # NOTE we have to write the pid inside the bash command to get the correct pid.
    cmd = 'nohup /darknet/darknet detector train data.txt tiny_yolo.cfg -dont_show -map >> last_training.log 2>&1 & echo $! > last_training.pid'
    p = subprocess.Popen(cmd, shell=True)
    _, err = p.communicate()
    if p.returncode != 0:
        raise Exception(f'Failed to start training with error: {err}')

    node.status.model['training_id'] = training_id


def _stop_training(training_id: str) -> None:
    training_path = get_training_path_by_id(training_id)
    os.chdir(training_path)
    cmd = 'kill -9 `cat last_training.pid`; rm last_training.pid'
    p = subprocess.Popen(cmd, shell=True)
    _, err = p.communicate()
    if p.returncode != 0:
        raise Exception(f'Failed to stop training with error: {err}')


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


def _check_state() -> None:
    if node.status.model:
        training_id = node.status.model['training_id']

        if training_id:
            state = get_training_state(training_id)
            if state == 'crashed':
                try:
                    _stop_training(training_id)
                except:
                    pass

                node.status.model = None
                node.status.state = State.Idle


def get_training_state(training_id):
    training_path = get_training_path_by_id(training_id)
    pid_path = f'{training_path}/last_training.pid'
    if not os.path.exists(pid_path):
        return 'stopped'
    with open(pid_path, 'r') as f:
        pid = f.read().strip()
    try:
        p = psutil.Process(int(pid))
    except psutil.NoSuchProcess as e:
        return 'crashed'
    if p.name() == 'darknet':
        return 'running'
    return 'crashed'


def get_training_path_by_id(trainings_id: str) -> str:
    trainings = [training_path for training_path in glob(
        f'/data/**/trainings/{trainings_id}', recursive=True)]
    return trainings[0]


@ node.on_event("shutdown")
async def shutdown():

    def restart():
        asyncio.create_task(node.sio.disconnect())

    Thread(target=restart).start()


# setting up backdoor_controls
node.include_router(backdoor_controls.router, prefix="")
