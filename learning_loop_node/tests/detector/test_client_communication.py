import asyncio
import json
import os

import pytest
import requests

from ...data_classes import ModelInformation
from ...detector.detector_node import DetectorNode
from ...globals import GLOBALS
from .conftest import get_outbox_files
from .testing_detector import TestingDetectorLogic

file_path = os.path.abspath(__file__)
test_image_path = os.path.join(os.path.dirname(file_path), 'test.jpg')


@pytest.mark.asyncio
async def test_detector_path(test_detector_node: DetectorNode):
    assert test_detector_node.outbox.path.startswith('/tmp')

# pylint: disable=unused-argument


async def test_sio_detect(test_detector_node, sio_client):
    with open(test_image_path, 'rb') as f:
        image_bytes = f.read()

    await asyncio.sleep(5)
    result = await sio_client.call('detect', {'image': image_bytes})
    assert len(result['box_detections']) == 1
    assert result['box_detections'][0]['category_name'] == 'some_category_name'
    assert result['box_detections'][0]['category_id'] == 'some_id'

    assert len(result['point_detections']) == 1
    assert result['point_detections'][0]['category_name'] == 'some_category_name_2'
    assert result['point_detections'][0]['category_id'] == 'some_id_2'

    assert len(result['segmentation_detections']) == 1
    assert result['segmentation_detections'][0]['category_name'] == 'some_category_name_3'
    assert result['segmentation_detections'][0]['category_id'] == 'some_id_3'

    assert len(result['classification_detections']) == 1
    assert result['classification_detections'][0]['category_name'] == 'some_category_name_4'
    assert result['classification_detections'][0]['category_id'] == 'some_id_4'


@pytest.mark.parametrize('grouping_key', ['mac', 'camera_id'])
def test_rest_detect(test_detector_node: DetectorNode, grouping_key: str):
    image = {('file', open(test_image_path, 'rb'))}
    headers = {grouping_key: '0:0:0:0', 'tags':  'some_tag'}

    assert isinstance(test_detector_node.detector_logic, TestingDetectorLogic)
    # test_detector_node.detector_logic.mock_is_initialized = True
    # print(test_detector_node.detector_logic.mock_is_initialized)
    # print(test_detector_node.detector_logic.is_initialized)
    response = requests.post(f'http://localhost:{GLOBALS.detector_port}/detect',
                             files=image, headers=headers, timeout=30)
    assert response.status_code == 200
    result = response.json()
    assert len(result['box_detections']) == 1
    assert result['box_detections'][0]['category_name'] == 'some_category_name'
    assert result['box_detections'][0]['category_id'] == 'some_id'

    assert len(result['point_detections']) == 1
    assert result['point_detections'][0]['category_name'] == 'some_category_name_2'
    assert result['point_detections'][0]['category_id'] == 'some_id_2'

    assert len(result['segmentation_detections']) == 1
    assert result['segmentation_detections'][0]['category_name'] == 'some_category_name_3'
    assert result['segmentation_detections'][0]['category_id'] == 'some_id_3'


def test_rest_upload(test_detector_node: DetectorNode):
    assert len(get_outbox_files(test_detector_node.outbox)) == 0

    image = {('files', open(test_image_path, 'rb'))}
    response = requests.post(f'http://localhost:{GLOBALS.detector_port}/upload', files=image, timeout=30)
    assert response.status_code == 200
    assert len(get_outbox_files(test_detector_node.outbox)) == 2, 'There should be one image and one .json file.'


@pytest.mark.parametrize('test_detector_node', [True], indirect=True)
async def test_sio_upload(test_detector_node: DetectorNode, sio_client):
    assert len(get_outbox_files(test_detector_node.outbox)) == 0

    with open(test_image_path, 'rb') as f:
        image_bytes = f.read()
    result = await sio_client.call('upload', {'image': image_bytes})
    assert result is None
    assert len(get_outbox_files(test_detector_node.outbox)) == 2, 'There should be one image and one .json file.'


# NOTE: This test seems to be flaky.
async def test_about_endpoint(test_detector_node: DetectorNode):
    await asyncio.sleep(11)
    response = requests.get(f'http://localhost:{GLOBALS.detector_port}/about', timeout=30)

    assert response.status_code == 200, response.content
    response_dict = json.loads(response.content)
    assert response_dict['model_info']
    model_information = ModelInformation.from_dict(response_dict['model_info'])

    assert response_dict['operation_mode'] == 'idle'
    assert response_dict['state'] == 'online'
    assert response_dict['target_model'] == '1.1'
    assert any(c.name == 'purple point' for c in model_information.categories)


async def test_model_version_api(test_detector_node: DetectorNode):
    await asyncio.sleep(11)

    response = requests.get(f'http://localhost:{GLOBALS.detector_port}/model_version', timeout=30)
    assert response.status_code == 200, response.content
    response_dict = json.loads(response.content)
    assert response_dict['version_control'] == 'follow_loop'
    assert response_dict['current_version'] == '1.1'
    assert response_dict['target_version'] == '1.1'
    assert response_dict['loop_version'] == '1.1'
    assert response_dict['local_versions'] == ['1.1']

    response = requests.put(f'http://localhost:{GLOBALS.detector_port}/model_version', data='1.0', timeout=30)
    assert response.status_code == 200, response.content
    response = requests.get(f'http://localhost:{GLOBALS.detector_port}/model_version', timeout=30)
    assert response.status_code == 200, response.content
    response_dict = json.loads(response.content)
    assert response_dict['version_control'] == 'specific_version'
    assert response_dict['current_version'] == '1.1'
    assert response_dict['target_version'] == '1.0'
    assert response_dict['loop_version'] == '1.1'
    assert response_dict['local_versions'] == ['1.1']

    await asyncio.sleep(11)
    response = requests.get(f'http://localhost:{GLOBALS.detector_port}/model_version', timeout=30)
    assert response.status_code == 200, response.content
    response_dict = json.loads(response.content)
    assert response_dict['version_control'] == 'specific_version'
    assert response_dict['current_version'] == '1.0'
    assert response_dict['target_version'] == '1.0'
    assert response_dict['loop_version'] == '1.1'
    assert set(response_dict['local_versions']) == set(['1.1', '1.0'])

    response = requests.put(f'http://localhost:{GLOBALS.detector_port}/model_version', data='pause', timeout=30)
    assert response.status_code == 200, response.content
    await asyncio.sleep(11)
    response = requests.get(f'http://localhost:{GLOBALS.detector_port}/model_version', timeout=30)
    assert response.status_code == 200, response.content
    response_dict = json.loads(response.content)
    assert response_dict['version_control'] == 'pause'
    assert response_dict['current_version'] == '1.0'
    assert response_dict['target_version'] == '1.0'
    assert response_dict['loop_version'] == '1.1'
    assert set(response_dict['local_versions']) == set(['1.1', '1.0'])

    response = requests.put(f'http://localhost:{GLOBALS.detector_port}/model_version', data='follow_loop', timeout=30)
    await asyncio.sleep(11)
    response = requests.get(f'http://localhost:{GLOBALS.detector_port}/model_version', timeout=30)
    assert response.status_code == 200, response.content
    response_dict = json.loads(response.content)
    assert response_dict['version_control'] == 'follow_loop'
    assert response_dict['current_version'] == '1.1'
    assert response_dict['target_version'] == '1.1'
    assert response_dict['loop_version'] == '1.1'
    assert set(response_dict['local_versions']) == set(['1.1', '1.0'])


async def test_rest_outbox_mode(test_detector_node: DetectorNode):
    await asyncio.sleep(3)

    def check_switch_to_mode(mode: str):
        response = requests.put(f'http://localhost:{GLOBALS.detector_port}/outbox_mode',
                                data=mode, timeout=30)
        assert response.status_code == 200, response.content
        response = requests.get(f'http://localhost:{GLOBALS.detector_port}/outbox_mode', timeout=30)
        assert response.status_code == 200, response.content
        assert response.content == mode.encode()

    check_switch_to_mode('stopped')
    check_switch_to_mode('continuous_upload')
    check_switch_to_mode('stopped')


async def test_api_responsive_during_large_upload(test_detector_node: DetectorNode):
    assert len(get_outbox_files(test_detector_node.outbox)) == 0

    with open(test_image_path, 'rb') as f:
        image_bytes = f.read()

    for _ in range(200):
        test_detector_node.outbox.save(image_bytes)

    outbox_size_early = len(get_outbox_files(test_detector_node.outbox))
    await asyncio.sleep(5)  # NOTE: we wait 5 seconds because the continuous upload is running every 5 seconds

    # check if api is still responsive
    response = requests.get(f'http://localhost:{GLOBALS.detector_port}/outbox_mode', timeout=2)
    assert response.status_code == 200, response.content

    await asyncio.sleep(5)
    outbox_size_late = len(get_outbox_files(test_detector_node.outbox))
    assert outbox_size_late > 0, 'The outbox should not be fully cleared, maybe the node was too fast.'
    assert outbox_size_early > outbox_size_late, 'The outbox should have been partially emptied.'
