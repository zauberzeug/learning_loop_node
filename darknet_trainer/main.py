from learning_loop_node.trainer.training_data import TrainingData
from learning_loop_node.trainer.trainer import Trainer
import uvicorn
import asyncio
import shutil
from log_parser import LogParser
from threading import Thread
import backdoor_controls
from fastapi_utils.tasks import repeat_every
import io
from learning_loop_node.node import Node
import learning_loop_node.node_helper as node_helper
import os
from typing import List, Union
import helper
import yolo_helper
import yolo_cfg_helper
from uuid import uuid4
import os
from glob import glob
import subprocess
from icecream import ic
import psutil
from status import Status
from uuid import uuid4
import traceback
import model_updater


node = Trainer(uuid='c34dc41f-9b76-4aa9-8b8d-9d27e33a19e4',
               name='darknet trainer')


@node.begin_training
async def begin_training(data: dict) -> None:
    try:
        training = node.training
        training_data = await download_data(node, data, training.images_folder, training.training_folder)

        await _prepare_training(node, training.training_folder, training.images_folder, training_data, training.id)
        await _start_training(training.id)
    except Exception as e:
        traceback.print_exc()
        raise e


async def download_data(node: Trainer, data: dict, image_folder, training_folder):
    loop = asyncio.get_event_loop()
    image_data_coroutine = node_helper.download_images_data(
        node.url, node.headers, node.training.organization, node.training.project,  data['image_ids'])

    image_data_task = loop.create_task(image_data_coroutine)
    urls, ids = node_helper.create_resource_urls(
        node.url, node.training.organization, node.training.project, filter_needed_image_ids(data['image_ids'], image_folder))
    await node_helper.download_images(loop, urls, ids, node.headers, image_folder)

    image_data = await image_data_task
    ic(f'Done downloading image_data for {len(image_data)} images.')
    node_helper.download_model(node.url, node.headers, training_folder, node.training.organization,
                               node.training.project, node.training.base_model.id)
    return TrainingData(image_data=image_data, box_categories=data['box_categories'])


def filter_needed_image_ids(all_image_ids, image_folder):
    ids = [os.path.splitext(os.path.basename(image))[0] for image in glob(f'{image_folder}/*.jpg')]
    return [id for id in all_image_ids if id not in ids]


async def _prepare_training(node: Node, training_folder, image_folder, training_data: TrainingData, training_uuid: str) -> None:
    yolo_helper.create_backup_dir(training_folder)

    image_folder_for_training = yolo_helper.create_image_links(
        training_folder, image_folder, training_data.image_ids())

    await yolo_helper.update_yolo_boxes(node, image_folder_for_training, training_data)

    box_category_names = helper.get_box_category_names(training_data)
    box_category_count = len(box_category_names)
    yolo_helper.create_names_file(training_folder, box_category_names)
    yolo_helper.create_data_file(training_folder, box_category_count)
    yolo_helper.create_train_and_test_file(
        training_folder, image_folder_for_training, training_data.image_data)

    yolo_cfg_helper.replace_classes_and_filters(
        box_category_count, training_folder)
    yolo_cfg_helper.update_anchors(training_folder)


async def _start_training(training_id: str) -> None:
    training_path = helper.get_training_path_by_id(training_id)

    weightfile = find_weightfile(training_path)
    cfg_file = find_cfg_file(training_path)
    # NOTE we have to write the pid inside the bash command to get the correct pid.
    cmd = f'cd {training_path};nohup /darknet/darknet detector train data.txt {cfg_file} {weightfile} -dont_show -map >> last_training.log 2>&1 & echo $! > last_training.pid'
    p = subprocess.Popen(cmd, shell=True)
    _, err = p.communicate()
    if p.returncode != 0:
        raise Exception(f'Failed to start training with error: {err}')


def find_weightfile(training_path: str) -> str:
    weightfiles = glob(f'{training_path}/*.weights', recursive=True)
    if not weightfiles or len(weightfiles) > 1:
        raise Exception('Number of present weightfiles must be 1.')
    weightfile = weightfiles[0].split('/')[-1]
    if weightfile == 'fake_weightfile.weights':
        return ""
    else:
        return weightfile


def find_cfg_file(training_path: str) -> str:
    cfg_files = weightfiles = glob(f'{training_path}/*.cfg', recursive=True)
    if not weightfiles or len(weightfiles) > 1:
        raise Exception(f'Number of present .cfg files must be 1, but was {len(weightfiles)}')
    cfg_file = cfg_files[0].split('/')[-1]
    return cfg_file


@node.stop_training
def stop() -> None:
    _stop_training(node, node.training.id)


async def _stop_training(node: Trainer, training_id: str) -> None:
    try:
        _kill_training_process(training_id)
    except Exception as e:
        print(e, flush=True)
    node.training = None

    new_status = Status(id=node.status.id, name=node.status.name)
    await node.update_status(new_status)


def _kill_training_process(training_id: str) -> None:
    training_path = helper.get_training_path_by_id(training_id)

    cmd = f'cd {training_path};kill -9 `cat last_training.pid`; rm last_training.pid'
    p = subprocess.Popen(cmd, shell=True)
    _, err = p.communicate()
    if p.returncode != 0:
        raise Exception(f'Failed to stop training with error: {err}')


@node.on_event("startup")
@repeat_every(seconds=5, raise_exceptions=False, wait_first=False)
async def check_state() -> None:
    """checking the current status"""
    try:
        await _check_state()
    except:
        traceback.print_exc()


async def _check_state() -> None:
    ic(f"checking state: {node.status.state}")

    if node.training:
        training_id = node.training.id
        ic(training_id,)
        await model_updater.check_state(training_id, node)
        await _check_training_state(node, training_id)


async def _check_training_state(node: Trainer, training_id: str) -> None:
    state = get_training_state(training_id)
    ic(state, )
    if state == 'crashed':
        await _stop_training(node, training_id)


def get_training_state(training_id):
    training_path = helper.get_training_path_by_id(training_id)
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


@node.get_model_files
def get_model_files(organization: str, project: str, model_id: str) -> List[str]:
    return _get_model_files(model_id)


def _get_model_files(model_id: str) -> List[str]:
    try:
        weightfile_path = glob(f'/data/**/trainings/**/{model_id}.weights', recursive=True)[0]
    except:
        raise Exception(f'No model found for id: {model_id}.')

    training_path = '/'.join(weightfile_path.split('/')[:-1])
    cfg_file_path = find_cfg_file(training_path)
    return [weightfile_path, f'{training_path}/{cfg_file_path}', f'{training_path}/names.txt']


@node.on_event("shutdown")
async def shutdown():

    def restart():
        asyncio.create_task(node.sio.disconnect())

    Thread(target=restart).start()


# setting up backdoor_controls
node.include_router(backdoor_controls.router, prefix="")

if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    uvicorn.run("main:node", host="0.0.0.0", port=80, lifespan='on', reload=True)
