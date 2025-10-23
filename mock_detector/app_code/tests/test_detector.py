import asyncio
import os

import numpy as np
import pytest
from PIL import Image

from learning_loop_node.globals import GLOBALS

# pylint: disable=unused-argument


file_path = os.path.abspath(__file__)
test_image_path = os.path.join(os.path.dirname(file_path), 'test.jpg')


@pytest.fixture(scope="session")
def event_loop():
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


@pytest.mark.usefixtures('test_detector_node')
async def test_sio_detect(sio):
    image = np.array(Image.open(test_image_path))

    response = await sio.call('detect', {'image': {'bytes': image.tobytes(), 'shape': image.shape, 'dtype': str(image.dtype)}})
    assert response['box_detections'] == []
    assert response['point_detections'] == []
    assert response['segmentation_detections'] == []
    assert response['tags'] == []
    assert 'created' in response
