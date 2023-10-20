import asyncio

import pytest

from learning_loop_node.loop_communication import LoopCommunicator


@pytest.fixture(autouse=True, scope='function')
async def create_project_for_module():

    loop_communicator = LoopCommunicator()
    await loop_communicator.delete("/zauberzeug/projects/pytest?keep_images=true")
    await asyncio.sleep(1)
    project_configuration = {
        'project_name': 'pytest', 'inbox': 0, 'annotate': 0, 'review': 0, 'complete': 3, 'image_style': 'beautiful',
        'box_categories': 2, 'point_categories': 2, 'segmentation_categories': 2, 'thumbs': False, 'tags': 0,
        'trainings': 1, 'box_detections': 3, 'box_annotations': 0}
    assert (await loop_communicator.post("/zauberzeug/projects/generator", json=project_configuration)).status_code == 200
    yield
    await loop_communicator.delete("/zauberzeug/projects/pytest?keep_images=true")
    await loop_communicator.shutdown()
