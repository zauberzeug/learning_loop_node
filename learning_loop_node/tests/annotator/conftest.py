import asyncio
import logging
import os
import shutil

import pytest

from ...globals import GLOBALS
from ...loop_communication import LoopCommunicator
from ..test_helper import ANNOTATOR_PROJECT, delete_project, project_name

PROJECT = project_name(ANNOTATOR_PROJECT)


@pytest.fixture()
async def setup_test_project():  # pylint: disable=redefined-outer-name
    loop_communicator = LoopCommunicator()
    await delete_project(loop_communicator, PROJECT, keep_images=True)
    await asyncio.sleep(1)
    project_conf = {
        'project_name': PROJECT, 'inbox': 0, 'annotate': 0, 'review': 0, 'complete': 3, 'image_style': 'beautiful',
        'box_categories': 2, 'point_categories': 2, 'segmentation_categories': 2, 'thumbs': False, 'tags': 0,
        'trainings': 1, 'box_detections': 3, 'box_annotations': 0}
    assert (await loop_communicator.post("/zauberzeug/projects/generator", json=project_conf)).status_code == 200
    yield
    try:
        await delete_project(loop_communicator, PROJECT, keep_images=True)
    finally:
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
