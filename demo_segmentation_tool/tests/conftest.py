import pytest
from typing import Generator
from learning_loop_node.tests import test_helper
from learning_loop_node.globals import GLOBALS
import shutil


@pytest.fixture()
def web() -> Generator:
    with test_helper.LiveServerSession() as c:
        yield c


@pytest.fixture(autouse=True, scope='module')
def create_project():
    test_helper.LiveServerSession().delete(f"/api/zauberzeug/projects/pytest?keep_images=true")
    project_configuration = {'project_name': 'pytest', 'inbox': 0, 'annotate': 0, 'review': 0, 'complete': 3, 'image_style': 'beautiful',
                             'box_categories': 2, 'segmentation_categories': 2, 'point_categories': 2, 'thumbs': False, 'tags': 0, 'trainings': 1, 'box_detections': 3, 'box_annotations': 0}
    assert test_helper.LiveServerSession().post(f"/api/zauberzeug/projects/generator",
                                                json=project_configuration).status_code == 200
    yield
    test_helper.LiveServerSession().delete(f"/api/zauberzeug/projects/pytest?keep_images=true")
