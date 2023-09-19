import asyncio

import pytest

from learning_loop_node.detector.detector_node import DetectorNode
from learning_loop_node.globals import GLOBALS


@pytest.fixture(scope="session")
def event_loop(request):
    """https://stackoverflow.com/a/66225169/4082686
       Create an instance of the default event loop for each test case.
       Prevents 'RuntimeError: Event loop is closed'
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


def test_assert_data_folder_for_tests():
    assert GLOBALS.data_folder != '/data'
    assert GLOBALS.data_folder.startswith('/tmp')


async def test_sio_detect(test_detector_node: DetectorNode, sio):
    with open('tests/test.jpg', 'rb') as f:
        image_bytes = f.read()

    response = await sio.call('detect', {'image': image_bytes})
    assert response['box_detections'] == []
    assert response['point_detections'] == []
    assert response['segmentation_detections'] == []
    assert response['tags'] == []
    assert 'date' in response
