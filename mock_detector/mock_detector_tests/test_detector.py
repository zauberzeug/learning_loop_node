from learning_loop_node.detector.operation_mode import OperationMode
from learning_loop_node.globals import GLOBALS
import pytest
from learning_loop_node.tests import test_helper
from main import detector_node
import os
from icecream import ic


@pytest.fixture()
def create_project():
    test_helper.LiveServerSession().delete(f"/api/zauberzeug/projects/pytest?keep_images=true")
    project_configuration = {'project_name': 'pytest', 'inbox': 0, 'annotate': 0, 'review': 0, 'complete': 0, 'image_style': 'plain',
                             'thumbs': False}
    assert test_helper.LiveServerSession().post(f"/api/zauberzeug/projects/generator",
                                                json=project_configuration).status_code == 200
    yield
    test_helper.LiveServerSession().delete(f"/api/zauberzeug/projects/pytest?keep_images=true")


@pytest.fixture(autouse=True, scope='function')
async def wait_for_sio():
    from icecream import ic
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
    model_id = await test_helper.assert_upload_model()

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
