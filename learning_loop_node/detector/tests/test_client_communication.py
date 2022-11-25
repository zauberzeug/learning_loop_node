import pytest
from learning_loop_node import DetectorNode
import requests
from learning_loop_node.detector.tests.conftest import get_outbox_files


def test_detector_path(test_detector_node: DetectorNode):
    assert test_detector_node.outbox.path.startswith('/tmp')


@pytest.mark.parametrize('test_detector_node', [True], indirect=True)
async def test_sio_detect(test_detector_node, sio_client):
    with open('detector/tests/test.jpg', 'rb') as f:
        image_bytes = f.read()
    result = await sio_client.call('detect', {'image': image_bytes})
    assert len(result['box_detections']) == 1
    assert result['box_detections'][0]['category_name'] == 'some_category_name'
    assert result['box_detections'][0]['category_id'] == 'some_id'

    assert len(result['point_detections']) == 1
    assert result['point_detections'][0]['category_name'] == 'some_category_name'
    assert result['point_detections'][0]['category_id'] == 'some_id'

    assert len(result['segmentation_detections']) == 1
    assert result['segmentation_detections'][0]['category_name'] == 'some_category_name'
    assert result['segmentation_detections'][0]['category_id'] == 'some_id'


@pytest.mark.parametrize('grouping_key', ['mac', 'camera_id'])
@pytest.mark.parametrize('test_detector_node', [True], indirect=True)
def test_rest_detect(test_detector_node: DetectorNode, grouping_key: str):
    image = {('file', open('detector/tests/test.jpg', 'rb'))}
    headers = {grouping_key: '0:0:0:0', 'tags':  'some_tag'}
    response = requests.post(f'http://localhost:{pytest.detector_port}/detect', files=image, headers=headers)
    assert response.status_code == 200
    result = response.json()
    assert len(result['box_detections']) == 1
    assert result['box_detections'][0]['category_name'] == 'some_category_name'
    assert result['box_detections'][0]['category_id'] == 'some_id'

    assert len(result['point_detections']) == 1
    assert result['point_detections'][0]['category_name'] == 'some_category_name'
    assert result['point_detections'][0]['category_id'] == 'some_id'

    assert len(result['segmentation_detections']) == 1
    assert result['segmentation_detections'][0]['category_name'] == 'some_category_name'
    assert result['segmentation_detections'][0]['category_id'] == 'some_id'


def test_rest_upload(test_detector_node: DetectorNode):
    assert len(get_outbox_files(test_detector_node.outbox)) == 0

    image = {('files', open('detector/tests/test.jpg', 'rb'))}
    response = requests.post(f'http://localhost:{pytest.detector_port}/upload', files=image)
    assert response.status_code == 200
    assert len(get_outbox_files(test_detector_node.outbox)) == 2, 'There should be one image and one .json file.'


@pytest.mark.parametrize('test_detector_node', [True], indirect=True)
async def test_sio_upload(test_detector_node: DetectorNode, sio_client):
    assert len(get_outbox_files(test_detector_node.outbox)) == 0

    with open('detector/tests/test.jpg', 'rb') as f:
        image_bytes = f.read()
    result = await sio_client.call('upload', {'image': image_bytes})
    assert result == None
    assert len(get_outbox_files(test_detector_node.outbox)) == 2, 'There should be one image and one .json file.'
