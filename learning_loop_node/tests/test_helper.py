import asyncio
import os
import shutil
import time
import zipfile
from glob import glob
from typing import Callable

import httpx
import pytest

from ..data_classes import (
    BoxDetection,
    ClassificationDetection,
    Detections,
    Point,
    PointDetection,
    SegmentationDetection,
    Shape,
)
from ..data_classes.image_metadata import ImageMetadata
from ..loop_communication import LoopCommunicator

TEST_PROJECT_SUFFIX = 'LOOP_TEST_PROJECT_SUFFIX'

# every project a suite creates and deletes on the loop, by its bare name -- the one place they are spelled out;
# a suite passes its name through project_name, and cleanup_test_projects sweeps exactly these
GENERAL_PROJECT = 'pytest_nodelib_general'
ANNOTATOR_PROJECT = 'pytest_nodelib_annotator'
MOCK_TRAINER_PROJECT_1 = 'pytest_mock_trainer_test1'
MOCK_TRAINER_PROJECT_2 = 'pytest_mock_trainer_test2'
TEST_PROJECTS = (GENERAL_PROJECT, ANNOTATOR_PROJECT, MOCK_TRAINER_PROJECT_1, MOCK_TRAINER_PROJECT_2)


def project_name(name: str) -> str:
    """The name of a project a suite creates and deletes on the shared loop.

    CI sets `LOOP_TEST_PROJECT_SUFFIX` to `<run id>_<run attempt>_<job index>`, so two runs at once -- two branches
    pushed close together -- each get their own project instead of deleting each other's, and `cleanup_test_projects`
    can tell from the name whether the run that created it is over. Locally the bare name is used."""
    suffix = os.environ.get(TEST_PROJECT_SUFFIX)
    return f'{name}_{suffix}' if suffix else name


async def delete_project(loop_communicator: LoopCommunicator, project: str, *, keep_images: bool = False,
                         timeout: float = 75) -> None:
    """Delete `zauberzeug/<project>` if it exists, or fail the test saying why.

    A fresh per-run project normally does not exist yet, so a quick lookup comes first and the delete is only sent
    for a project that is really there. A delete on the loop can take up to a minute before it answers, so the
    default timeout outlasts that and a failure carries the loop's own answer instead of a bare ReadTimeout."""
    path = f'/zauberzeug/projects/{project}'
    try:
        response = await loop_communicator.get(path, timeout=10)
        if response.status_code == 404:
            return
        response = await loop_communicator.delete(path + ('?keep_images=true' if keep_images else ''), timeout=timeout)
    except httpx.TimeoutException:
        pytest.fail(f'deleting {path} on {loop_communicator.host} took longer than {timeout} s', pytrace=False)
    if response.status_code not in (200, 404):
        pytest.fail(f'deleting {path} on {loop_communicator.host} answered {response.status_code}: {response.text}',
                    pytrace=False)


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


def get_dummy_detections() -> Detections:
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


def get_dummy_metadata() -> ImageMetadata:
    detections = get_dummy_detections()
    return ImageMetadata(box_detections=detections.box_detections,
                         point_detections=detections.point_detections,
                         segmentation_detections=detections.segmentation_detections,
                         classification_detections=detections.classification_detections)
