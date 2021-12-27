from typing import Generator
from multiprocessing import Process
import pytest
import socketio
import asyncio
import logging
import uvicorn
from learning_loop_node.detector.detector_node import DetectorNode
import requests
from icecream import ic
import json
from testing_detector import TestingDetector
from learning_loop_node.globals import GLOBALS
import os
from glob import glob

port = 5001


@pytest.fixture()
async def test_detector_node():
    os.environ['ORGANIZATION'] = 'zauberzeug'
    os.environ['PROJECT'] = 'demo'

    det = TestingDetector()
    node = DetectorNode(name='test', detector=det)

    proc = Process(target=uvicorn.run,
                   args=(node,),
                   kwargs={
                       "host": "127.0.0.1",
                       "port": port,
                   },

                   daemon=True)
    proc.start()
    await asyncio.sleep(5)  # time for the server to start
    yield node
    await node.sio_client.disconnect()
    proc.terminate()


@pytest.fixture()
async def sio_client() -> Generator:
    sio = socketio.AsyncClient()
    try_connect = True
    retry_count = 0
    while try_connect:
        try:
            await sio.connect(f"ws://localhost:{port}", socketio_path="/ws/socket.io")
            try_connect = False
        except:
            logging.warning('trying again')
            await asyncio.sleep(1)
        retry_count += 1
        if retry_count > 10:
            raise Exception('Max Retry')

    yield sio
    await sio.disconnect()


def test_rest_detect(test_detector_node: DetectorNode):
    image = {('file', open('detector/tests/test.jpg', 'rb'))}
    headers = {'mac': '0:0:0:0', 'tags':  'some_tag'}
    request = requests.post(f'http://localhost:{port}/detect', files=image, headers=headers)
    assert request.status_code == 200

    json_content = json.loads(request.content.decode('utf-8'))
    assert len(json_content['box_detections']) == 1
    assert json_content['box_detections'][0]['category_name'] == 'some_category_name'


@pytest.mark.asyncio
async def test_sio_detect(test_detector_node: DetectorNode, sio_client):
    with open('detector/tests/test.jpg', 'rb') as f:
        image_bytes = f.read()

    result = await sio_client.call('detect', {'image': image_bytes})

    assert len(result['box_detections']) == 1
    assert result['box_detections'][0]['category_name'] == 'some_category_name'


def test_rest_upload(test_detector_node: DetectorNode):
    assert test_detector_node.outbox.path.startswith('/tmp')

    def get_outbox_files():
        files = glob(f'{test_detector_node.outbox.path}/**/*', recursive=True)
        return [file for file in files if os.path.isfile(file)]

    assert len(get_outbox_files()) == 0
    image = {('files', open('detector/tests/test.jpg', 'rb'))}
    request = requests.post(f'http://localhost:{port}/upload', files=image)
    assert request.status_code == 200
    ic(get_outbox_files())
    assert len(get_outbox_files()) == 2, 'There should be one image and one .json file.'


@pytest.mark.asyncio
async def test_sio_upload(test_detector_node: DetectorNode, sio_client):
    assert test_detector_node.outbox.path.startswith('/tmp')

    def get_outbox_files():
        files = glob(f'{test_detector_node.outbox.path}/**/*', recursive=True)
        return [file for file in files if os.path.isfile(file)]

    assert len(get_outbox_files()) == 0

    with open('detector/tests/test.jpg', 'rb') as f:
        image_bytes = f.read()
    result = await sio_client.call('upload', {'image': image_bytes})
    assert result == None

    assert len(get_outbox_files()) == 2, 'There should be one image and one .json file.'
