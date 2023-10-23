
import asyncio
import logging
import os
import shutil

import pytest

from learning_loop_node.data_classes import (BoxDetection,
                                             ClassificationDetection, Context,
                                             Detections, Point, PointDetection,
                                             SegmentationDetection, Shape)

from .data_exchanger import DataExchanger
from .globals import GLOBALS
from .loop_communication import LoopCommunicator


@pytest.fixture()
async def glc():
    loop_communicator = LoopCommunicator()
    yield loop_communicator
    await loop_communicator.shutdown()


@pytest.fixture()
async def data_exchanger():
    loop_communicator = LoopCommunicator()
    context = Context(organization='zauberzeug', project='pytest')
    dx = DataExchanger(context, loop_communicator)
    yield dx
    await loop_communicator.shutdown()


@pytest.fixture()
async def setup_test_project():  # pylint: disable=redefined-outer-name
    loop_communicator = LoopCommunicator()
    await loop_communicator.delete("/zauberzeug/projects/pytest_p?keep_images=true")
    await asyncio.sleep(1)
    project_conf = {
        'project_name': 'pytest_p', 'inbox': 0, 'annotate': 0, 'review': 0, 'complete': 3, 'image_style': 'beautiful',
        'box_categories': 2, 'point_categories': 2, 'segmentation_categories': 2, 'thumbs': False, 'tags': 0,
        'trainings': 1, 'box_detections': 3, 'box_annotations': 0}
    assert (await loop_communicator.post("/zauberzeug/projects/generator", json=project_conf)).status_code == 200
    yield
    await loop_communicator.delete("/zauberzeug/projects/pytest_p?keep_images=true")
    await loop_communicator.shutdown()


@pytest.fixture(autouse=True, scope='session')
def clear_loggers():
    """Remove handlers from all loggers"""
    # see https://github.com/pytest-dev/pytest/issues/5502
    yield

    loggers = [logging.getLogger()] + list(logging.Logger.manager.loggerDict.values())
    for logger in loggers:
        if not isinstance(logger, logging.Logger):
            continue
        handlers = getattr(logger, 'handlers', [])
        for handler in handlers:
            logger.removeHandler(handler)


@pytest.fixture(autouse=True, scope='function')
def data_folder():
    GLOBALS.data_folder = '/tmp/learning_loop_lib_data'
    shutil.rmtree(GLOBALS.data_folder, ignore_errors=True)
    os.makedirs(GLOBALS.data_folder, exist_ok=True)
    yield
    shutil.rmtree(GLOBALS.data_folder, ignore_errors=True)


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
