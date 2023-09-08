import asyncio

import pytest
import requests

from learning_loop_node import DetectorNode
from learning_loop_node.detector.tests.conftest import get_outbox_files
from learning_loop_node.globals import GLOBALS

from .testing_detector import TestingDetectorLogic


@pytest.mark.asyncio
async def test_detector_path(test_detector_node: DetectorNode):
    assert test_detector_node.outbox.path.startswith('/tmp')

# pylint: disable=unused-argument


async def test_sio_detect(test_detector_node, sio_client):
    with open('detector/tests/test.jpg', 'rb') as f:
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
    image = {('file', open('detector/tests/test.jpg', 'rb'))}
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

    image = {('files', open('detector/tests/test.jpg', 'rb'))}
    response = requests.post(f'http://localhost:{GLOBALS.detector_port}/upload', files=image, timeout=30)
    assert response.status_code == 200
    assert len(get_outbox_files(test_detector_node.outbox)) == 2, 'There should be one image and one .json file.'


@pytest.mark.parametrize('test_detector_node', [True], indirect=True)
async def test_sio_upload(test_detector_node: DetectorNode, sio_client):
    assert len(get_outbox_files(test_detector_node.outbox)) == 0

    with open('detector/tests/test.jpg', 'rb') as f:
        image_bytes = f.read()
    result = await sio_client.call('upload', {'image': image_bytes})
    assert result is None
    assert len(get_outbox_files(test_detector_node.outbox)) == 2, 'There should be one image and one .json file.'
