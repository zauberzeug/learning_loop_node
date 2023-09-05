
import logging
import os
import shutil

import pytest

from learning_loop_node.data_classes import Context

from .data_exchanger import DataExchanger
from .globals import GLOBALS
from .loop_communication import LoopCommunicator


@pytest.fixture()
async def glc():
    loop_communicator = LoopCommunicator()
    yield loop_communicator
    await loop_communicator.shutdown()


@pytest.mark.asyncio
@pytest.fixture()
async def data_downloader(glc):
    context = Context(organization='zauberzeug', project='pytest')
    dc = DataExchanger(context, glc)
    return dc


@pytest.mark.asyncio
@pytest.fixture()
async def setup_test_project(glc):  # pylint: disable=redefined-outer-name
    lc = glc
    assert (await lc.delete(
        "/zauberzeug/projects/pytest?keep_images=true")).status_code == 200
    project_conf = {
        'project_name': 'pytest', 'inbox': 0, 'annotate': 0, 'review': 0, 'complete': 3, 'image_style': 'beautiful',
        'box_categories': 2, 'point_categories': 2, 'segmentation_categories': 2, 'thumbs': False, 'tags': 0,
        'trainings': 1, 'box_detections': 3, 'box_annotations': 0}
    assert (await lc.post(
        "/zauberzeug/projects/generator", json=project_conf)).status_code == 200
    yield
    await lc.delete("/zauberzeug/projects/pytest?keep_images=true")


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
