import logging
from learning_loop_node.tests import test_helper
from learning_loop_node.loop import loop
import pytest
import shutil
import asyncio
import icecream

icecream.install()
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture()
def create_project():
    test_helper.LiveServerSession().delete(f"/api/zauberzeug/projects/pytest?keep_images=true")
    project_configuration = {'project_name': 'pytest', 'inbox': 0, 'annotate': 0, 'review': 0, 'complete': 3, 'image_style': 'beautiful',
                             'categories': 2, 'thumbs': False, 'tags': 0, 'trainings': 1, 'detections': 3, 'annotations': 0, 'skeleton': False}
    assert test_helper.LiveServerSession().post(f"/api/zauberzeug/projects/generator",
                                                json=project_configuration).status_code == 200
    yield
    test_helper.LiveServerSession().delete(f"/api/zauberzeug/projects/pytest?keep_images=true")


@pytest.fixture(scope='function')
def data_folder():
    path = '/tmp/learning_loop_lib_data'
    shutil.rmtree(path, ignore_errors=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True, scope='function')
def loop_session():
    loop.session = None
    yield
    loop.session = None
