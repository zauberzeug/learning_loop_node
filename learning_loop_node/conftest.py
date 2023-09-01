
import logging
import os
import shutil

import icecream
import pytest

from learning_loop_node.globals import GLOBALS
from learning_loop_node.loop_communication import glc
from learning_loop_node.tests import test_helper

icecream.install()
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture()
def setup_test_project():
    test_helper.LiveServerSession().delete("/zauberzeug/projects/pytest?keep_images=true")
    project_configuration = {
        'project_name': 'pytest', 'inbox': 0, 'annotate': 0, 'review': 0, 'complete': 3, 'image_style': 'beautiful',
        'box_categories': 2, 'point_categories': 2, 'segmentation_categories': 2, 'thumbs': False, 'tags': 0,
        'trainings': 1, 'box_detections': 3, 'box_annotations': 0}
    assert test_helper.LiveServerSession().post("/zauberzeug/projects/generator",
                                                json=project_configuration).status_code == 200
    yield
    test_helper.LiveServerSession().delete("/zauberzeug/projects/pytest?keep_images=true")


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
async def loop_session():
    await glc.shutdown()
    yield
    await glc.shutdown()


@pytest.fixture(autouse=True, scope='function')
def data_folder():
    GLOBALS.data_folder = '/tmp/learning_loop_lib_data'
    shutil.rmtree(GLOBALS.data_folder, ignore_errors=True)
    os.makedirs(GLOBALS.data_folder, exist_ok=True)
    yield
    shutil.rmtree(GLOBALS.data_folder, ignore_errors=True)
