import asyncio
import logging
import os
import shutil

import pytest

from ...data_classes import Context
from ...data_exchanger import DataExchanger
from ...globals import GLOBALS
from ...loop_communication import LoopCommunicator


@pytest.fixture(autouse=True, scope='function')
async def create_project_for_module():

    loop_communicator = LoopCommunicator()
    await loop_communicator.delete("/zauberzeug/projects/pytest_nodelib_general?keep_images=true")
    await asyncio.sleep(1)
    project_configuration = {
        'project_name': 'pytest_nodelib_general', 'inbox': 0, 'annotate': 0, 'review': 0, 'complete': 3, 'image_style': 'beautiful',
        'box_categories': 2, 'point_categories': 2, 'segmentation_categories': 2, 'thumbs': False, 'tags': 0,
        'trainings': 1, 'box_detections': 3, 'box_annotations': 0}
    assert (await loop_communicator.post("/zauberzeug/projects/generator", json=project_configuration)).status_code == 200
    yield
    await loop_communicator.delete("/zauberzeug/projects/pytest_nodelib_general?keep_images=true")
    await loop_communicator.shutdown()


@pytest.fixture()
async def data_exchanger():
    loop_communicator = LoopCommunicator()
    context = Context(organization='zauberzeug', project='pytest_nodelib_general')
    dx = DataExchanger(context, loop_communicator)
    yield dx
    await loop_communicator.shutdown()

# ====================================== REDUNDANT FIXTURES IN ALL CONFTESTS ! ======================================


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
