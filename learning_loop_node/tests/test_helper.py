import asyncio
import os
import shutil
import time
import zipfile
from collections.abc import Callable
from glob import glob

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
DELETE_TIMEOUT = 75  # outlasts the loop's own 60 s limit, so its answer comes through

# the projects the suites create and delete on the loop; cleanup_test_projects sweeps exactly these
GENERAL_PROJECT = 'pytest_nodelib_general'
ANNOTATOR_PROJECT = 'pytest_nodelib_annotator'
MOCK_TRAINER_PROJECT_1 = 'pytest_mock_trainer_test1'
MOCK_TRAINER_PROJECT_2 = 'pytest_mock_trainer_test2'
TEST_PROJECTS = (GENERAL_PROJECT, ANNOTATOR_PROJECT, MOCK_TRAINER_PROJECT_1, MOCK_TRAINER_PROJECT_2)


def project_name(name: str) -> str:
    """Append `LOOP_TEST_PROJECT_SUFFIX`, which CI sets per job, so concurrent runs never share a project."""
    suffix = os.environ.get(TEST_PROJECT_SUFFIX)
    return f'{name}_{suffix}' if suffix else name


_timed_out_deletes: set[str] = set()


async def delete_project(loop_communicator: LoopCommunicator, project: str, *, keep_images: bool = False) -> None:
    """
    Delete `zauberzeug/<project>` if it exists, or fail the test saying why.

    The loop goes on deleting after a timeout, so a later call for the same project fails at once instead of sending
    a delete that would block on it. `keep_images` only applies to a bare name; a suffixed one is never reused.
    """
    path = f'/zauberzeug/projects/{project}'
    if project in _timed_out_deletes:
        pytest.fail(f'deleting {path} on {loop_communicator.host} timed out before, not sending another delete',
                    pytrace=False)
    keep_images = keep_images and not os.environ.get(TEST_PROJECT_SUFFIX)
    try:
        response = await loop_communicator.delete(path + ('?keep_images=true' if keep_images else ''),
                                                  timeout=DELETE_TIMEOUT)
    except httpx.TimeoutException:
        _timed_out_deletes.add(project)
        pytest.fail(f'deleting {path} on {loop_communicator.host} took longer than {DELETE_TIMEOUT} s', pytrace=False)
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
