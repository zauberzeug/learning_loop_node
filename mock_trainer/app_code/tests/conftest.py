import asyncio

import pytest

from learning_loop_node.loop_communication import LoopCommunicator
from learning_loop_node.testing import assert_not_production_loop
from learning_loop_node.testing.fixtures import clear_loggers, data_folder  # noqa: F401

# pylint: disable=redefined-outer-name


@pytest.fixture()
async def glc():
    loop_communicator = LoopCommunicator()
    yield loop_communicator
    await loop_communicator.shutdown()


@pytest.fixture()
async def setup_test_project1(glc: LoopCommunicator):
    assert_not_production_loop()
    await glc.delete("/zauberzeug/projects/pytest_mock_trainer_test1?keep_images=true")
    await asyncio.sleep(1)
    project_configuration = {
        'project_name': 'pytest_mock_trainer_test1', 'inbox': 1, 'annotate': 2, 'review': 3, 'complete': 4, 'image_style': 'plain',
        'box_categories': 1, 'point_categories': 1, 'segmentation_categories': 1, 'thumbs': False, 'trainings': 1}
    assert (await glc.post("/zauberzeug/projects/generator", json=project_configuration)).status_code == 200
    await asyncio.sleep(1)
    yield
    await glc.delete("/zauberzeug/projects/pytest_mock_trainer_test1?keep_images=true")
    await asyncio.sleep(1)


@pytest.fixture()
async def setup_test_project2(glc: LoopCommunicator):
    assert_not_production_loop()
    await glc.delete("/zauberzeug/projects/pytest_mock_trainer_test2?keep_images=true")
    await asyncio.sleep(1)
    project_configuration = {
        'project_name': 'pytest_mock_trainer_test2', 'inbox': 0, 'annotate': 0, 'review': 0, 'complete': 3, 'image_style': 'plain',
        'box_categories': 2, 'segmentation_categories': 2, 'point_categories': 2, 'thumbs': False, 'tags': 0,
        'trainings': 1, 'box_detections': 3, 'box_annotations': 0}
    assert (await glc.post("/zauberzeug/projects/generator", json=project_configuration)).status_code == 200
    await asyncio.sleep(1)
    yield
    await glc.delete("/zauberzeug/projects/pytest_mock_trainer_test2?keep_images=true")
    await asyncio.sleep(1)
