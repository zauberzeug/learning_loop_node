from learning_loop_node.globals import GLOBALS
from typing import Generator
import pytest
import asyncio
import logging
import socketio


@pytest.fixture(scope="session")
def event_loop(request):
    """https://stackoverflow.com/a/66225169/4082686
       Create an instance of the default event loop for each test case.
       Prevents 'RuntimeError: Event loop is closed'
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture()
async def sio() -> Generator:
    sio = socketio.AsyncClient()
    try_connect = True
    count = 0
    while try_connect:
        count += 1
        try:
            await sio.connect("ws://localhost", socketio_path="/ws/socket.io", wait_timeout=1)
            try_connect = False
        except:
            logging.warning('trying again')
            await asyncio.sleep(1)
        if count == 5:
            raise Exception('Could not connect to sio')

    assert sio.transport() == 'websocket'
    yield sio
    await sio.disconnect()


def test_assert_data_folder_for_tests():
    assert GLOBALS.data_folder != '/data'
    assert GLOBALS.data_folder.startswith('/tmp')


@pytest.mark.asyncio
async def test_sio_detect(sio):
    with open('mock_detector_tests/test.jpg', 'rb') as f:
        image_bytes = f.read()

    response = await sio.call('detect', {'image': image_bytes})
    assert response['box_detections'] == []
    assert response['point_detections'] == []
    assert response['segmentation_detections'] == []
    assert response['tags'] == []
    assert 'date' in response
