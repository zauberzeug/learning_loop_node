
import asyncio

import pytest

from learning_loop_node.loop_communication import LoopCommunicator


@pytest.fixture()
async def glc():
    loop_communicator = LoopCommunicator()
    yield loop_communicator
    await loop_communicator.shutdown()


@pytest.fixture(autouse=True, scope='function')
async def setup_test_project(glc: LoopCommunicator):
    await glc.delete("/zauberzeug/projects/pytest?keep_images=true")
    await asyncio.sleep(1)
    project_configuration = {
        'project_name': 'pytest', 'inbox': 0, 'annotate': 0, 'review': 0, 'complete': 3, 'image_style': 'beautiful',
        'box_categories': 2, 'segmentation_categories': 2, 'point_categories': 2, 'thumbs': False, 'tags': 0,
        'trainings': 1, 'box_detections': 3, 'box_annotations': 0}
    assert (await glc.post("/zauberzeug/projects/generator", json=project_configuration)).status_code == 200
    yield
    await glc.delete("/zauberzeug/projects/pytest?keep_images=true")
