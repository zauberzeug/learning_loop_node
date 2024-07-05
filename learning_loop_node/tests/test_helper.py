import asyncio
import os
import shutil
import time
import zipfile
from glob import glob
from typing import Callable

from ..data_classes import (BoxDetection, ClassificationDetection, Context, Detections, Point, PointDetection,
                            SegmentationDetection, Shape)
from ..helpers.misc import create_image_folder, create_project_folder, create_training_folder
from ..loop_communication import LoopCommunicator


def get_files_in_folder(folder: str):
    files = [entry for entry in glob(f'{folder}/**/*', recursive=True) if os.path.isfile(entry)]
    files.sort()
    return files


async def get_latest_model_id(project: str) -> str:
    lc = LoopCommunicator()
    response = await lc.get(f'/zauberzeug/projects/{project}/trainings')
    await lc.shutdown()

    assert response.status_code == 200
    trainings = response.json()
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


def get_dummy_detections():
    return Detections(
        box_detections=[
            BoxDetection(category_name='some_category_name', x=1, y=2, height=3, width=4,
                         model_name='some_model', confidence=.42, category_id='some_id')],
        point_detections=[
            PointDetection(category_name='some_category_name_2', x=10, y=12,
                           model_name='some_model', confidence=.42, category_id='some_id_2')],
        segmentation_detections=[
            SegmentationDetection(category_name='some_category_name_3',
                                  shape=Shape(points=[Point(x=1, y=1)]),
                                  model_name='some_model', confidence=.42,
                                  category_id='some_id_3')],
        classification_detections=[
            ClassificationDetection(category_name='some_category_name_4', model_name='some_model',
                                    confidence=.42, category_id='some_id_4')])
