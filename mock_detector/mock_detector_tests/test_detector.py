from learning_loop_node.detector.detector_node import DetectorNode
from learning_loop_node.detector.operation_mode import OperationMode
from learning_loop_node.globals import GLOBALS
import pytest
from learning_loop_node.tests import test_helper
from main import detector_node
import os
from icecream import ic
import asyncio
import main
from uuid import uuid4
from learning_loop_node.trainer.model import Model
import json
from importlib import reload


@pytest.fixture()
def create_project():
    test_helper.LiveServerSession().delete(f"/api/zauberzeug/projects/pytest?keep_images=true")
    project_configuration = {'project_name': 'pytest', 'inbox': 0, 'annotate': 0, 'review': 0, 'complete': 0, 'image_style': 'plain',
                             'thumbs': False}
    assert test_helper.LiveServerSession().post(f"/api/zauberzeug/projects/generator",
                                                json=project_configuration).status_code == 200
    yield
    test_helper.LiveServerSession().delete(f"/api/zauberzeug/projects/pytest?keep_images=true")


@pytest.fixture(scope="session")
def event_loop(request):
    """https://stackoverflow.com/a/66225169/4082686
       Create an instance of the default event loop for each test case.
       Prevents 'RuntimeError: Event loop is closed'
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True, scope='function')
async def wait_for_sio():
    organization = detector_node.organization
    project = detector_node.project

    detector_node.organization = 'zauberzeug'
    detector_node.project = 'pytest'
    await detector_node.sio_client.disconnect()

    await detector_node.connect()
    assert detector_node.sio_client.connected

    yield
    detector_node.organization = organization
    detector_node.project = project
    await detector_node.sio_client.disconnect()
    await detector_node.connect()
    assert detector_node.sio_client.connected


def test_assert_data_folder_for_tests():
    assert GLOBALS.data_folder != '/data'
    assert GLOBALS.data_folder.startswith('/tmp')


@pytest.mark.asyncio
async def test_mode_check_for_update(create_project):
    new_model = Model(
        id=str(uuid4()),
    )
    ic(new_model)
    response = await detector_node.sio_client.call('update_model', ('zauberzeug', 'pytest', new_model.__dict__))
    assert response == True

    model_id = await test_helper.assert_upload_model_with_id(model_id=new_model.id)
    assert new_model.id == model_id

    assert detector_node.sio_client.connected
    assert detector_node.operation_mode == OperationMode.Idle
    assert detector_node.current_model_id == None
    assert detector_node.target_model_id == None

    response = await detector_node.set_operation_mode(OperationMode.Check_for_updates)
    assert response == None

    # Update target model on loop (normally done by frontend)
    response = await detector_node.sio_client.call('update_deployment', {'target_model_id': model_id}, timeout=1)
    assert response == True

    await detector_node.check_for_update()
    assert detector_node.target_model_id == model_id
    assert os.path.exists(f'{GLOBALS.data_folder}/models/{model_id}/file_2.txt')
    assert os.path.exists(f'{GLOBALS.data_folder}/models/{model_id}/file_1.txt')
    assert os.path.exists(f'{GLOBALS.data_folder}/model/file_1.txt')
    assert os.path.exists(f'{GLOBALS.data_folder}/model/file_2.txt')

    # NOTE we cant really ensure, that the Node is restarted. Reloading the module creates a new instance of detector_node.
    reload(main)

    assert main.detector_node.current_model_id == model_id


@pytest.mark.asyncio
async def test_read_model_id_on_construction():
    model_id = 42

    node = DetectorNode(name='1')
    assert node.current_model_id == None
    os.makedirs(f'{GLOBALS.data_folder}/model')
    with open(f'{GLOBALS.data_folder}/model/model.json', 'w') as f:
        json.dump({'id': model_id}, f)

    node = DetectorNode(name='2')
    assert node.current_model_id == model_id
