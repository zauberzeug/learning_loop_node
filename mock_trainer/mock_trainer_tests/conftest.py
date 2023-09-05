import shutil

import pytest

from learning_loop_node.globals import GLOBALS
from learning_loop_node.loop_communication import LoopCommunicator


@pytest.fixture(autouse=True, scope='function')
def data_folder():
    GLOBALS.data_folder = '/tmp/learning_loop_lib_data'
    shutil.rmtree(GLOBALS.data_folder, ignore_errors=True)
    yield
    shutil.rmtree(GLOBALS.data_folder, ignore_errors=True)


@pytest.fixture(autouse=True, scope='module')
async def setup_test_project(glc: LoopCommunicator):
    await glc.delete("/zauberzeug/projects/pytest?keep_images=true")
    project_configuration = {
        'project_name': 'pytest', 'inbox': 0, 'annotate': 0, 'review': 0, 'complete': 3, 'image_style': 'plain',
        'box_categories': 2, 'segmentation_categories': 2, 'point_categories': 2, 'thumbs': False, 'tags': 0,
        'trainings': 1, 'box_detections': 3, 'box_annotations': 0}
    r = await glc.post("/zauberzeug/projects/generator", json=project_configuration)
    assert r.status_code == 200
    yield
    await glc.delete("/zauberzeug/projects/pytest?keep_images=true")
