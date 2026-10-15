import asyncio
import shutil

import pytest

from learning_loop_node.globals import GLOBALS
from learning_loop_node.loop_communication import LoopCommunicator
from learning_loop_node.tests.test_helper import delete_project, project_name

PROJECT1 = project_name('pytest_mock_trainer_test1')
PROJECT2 = project_name('pytest_mock_trainer_test2')

# pylint: disable=redefined-outer-name


@pytest.fixture()
async def glc():
    loop_communicator = LoopCommunicator()
    yield loop_communicator
    await loop_communicator.shutdown()


@pytest.fixture()
async def setup_test_project1(glc: LoopCommunicator):
    await delete_project(glc, PROJECT1, keep_images=True)
    await asyncio.sleep(1)
    project_configuration = {
        'project_name': PROJECT1, 'inbox': 1, 'annotate': 2, 'review': 3, 'complete': 4, 'image_style': 'plain',
        'box_categories': 1, 'point_categories': 1, 'segmentation_categories': 1, 'thumbs': False, 'trainings': 1}
    assert (await glc.post("/zauberzeug/projects/generator", json=project_configuration)).status_code == 200
    await asyncio.sleep(1)
    yield
    await delete_project(glc, PROJECT1, keep_images=True)
    await asyncio.sleep(1)


@pytest.fixture(autouse=True, scope='function')
def data_folder():
    GLOBALS.data_folder = '/tmp/learning_loop_lib_data'
    shutil.rmtree(GLOBALS.data_folder, ignore_errors=True)
    yield
    shutil.rmtree(GLOBALS.data_folder, ignore_errors=True)


@pytest.fixture()
async def setup_test_project2(glc: LoopCommunicator):
    await delete_project(glc, PROJECT2, keep_images=True)
    await asyncio.sleep(1)
    project_configuration = {
        'project_name': PROJECT2, 'inbox': 0, 'annotate': 0, 'review': 0, 'complete': 3, 'image_style': 'plain',
        'box_categories': 2, 'segmentation_categories': 2, 'point_categories': 2, 'thumbs': False, 'tags': 0,
        'trainings': 1, 'box_detections': 3, 'box_annotations': 0}
    assert (await glc.post("/zauberzeug/projects/generator", json=project_configuration)).status_code == 200
    await asyncio.sleep(1)
    yield
    await delete_project(glc, PROJECT2, keep_images=True)
    await asyncio.sleep(1)
