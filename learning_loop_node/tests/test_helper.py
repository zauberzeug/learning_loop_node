import asyncio
import os
import shutil
import time
import zipfile
from glob import glob
from typing import Callable

from learning_loop_node import node_helper
from learning_loop_node.data_classes import Context
from learning_loop_node.node import Node
from learning_loop_node.trainer.trainer import Trainer

from ..loop_communication import LoopCommunicator


def get_files_in_folder(folder: str):
    files = [entry for entry in glob(f'{folder}/**/*', recursive=True) if os.path.isfile(entry)]
    files.sort()
    return files


async def get_latest_model_id() -> str:
    lc = LoopCommunicator()
    response = await lc.get('/zauberzeug/projects/pytest/trainings')
    await lc.shutdown()

    assert response.status_code == 200
    trainings = response.json()
    # print(trainings) P? es gibt hier keine charts
    return trainings['charts'][0]['data'][0]['model_id']


def unzip(file_path, target_folder):
    shutil.rmtree(target_folder, ignore_errors=True)
    os.makedirs(target_folder)
    with zipfile.ZipFile(file_path, 'r') as zip_:
        zip_.extractall(target_folder)


async def condition(c_condition: Callable, *, timeout: float = 1.0, interval: float = 0.1):
    start = time.time()
    while not c_condition():
        if time.time() > start + timeout:
            raise TimeoutError(f'condition {c_condition} took longer than {timeout}s')
        await asyncio.sleep(interval)


def update_attributes(obj, **kwargs) -> None:
    if isinstance(obj, dict):
        _update_attribute_dict(obj, **kwargs)
    else:
        _update_attribute_class_instance(obj, **kwargs)


def _update_attribute_class_instance(obj, **kwargs) -> None:
    for key, value in kwargs.items():
        if hasattr(obj, key):
            setattr(obj, key, value)
        else:
            raise ValueError(f"Object of type '{type(obj)}' does not have a property '{key}'.")


def _update_attribute_dict(obj: dict, **kwargs) -> None:
    for key, value in kwargs.items():
        obj[key] = value

# TODO p?


def create_needed_folders(base_folder: str, training_uuid: str = 'some_uuid'):  # pylint: disable=unused-argument
    project_folder = Node.create_project_folder(
        Context(organization='zauberzeug', project='pytest'))
    image_folder = node_helper.create_image_folder(project_folder)
    training_folder = Trainer.create_training_folder(project_folder, training_uuid)
    return project_folder, image_folder, training_folder
