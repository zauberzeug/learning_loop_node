import asyncio
import logging
import multiprocessing
import os
import shutil
import socket
from multiprocessing import Process
from typing import AsyncGenerator

import pytest
import socketio
import uvicorn

from learning_loop_node.data_classes import Category, ModelInformation
from learning_loop_node.detector.detector_node import DetectorNode
from learning_loop_node.globals import GLOBALS

from ..mock_detector import MockDetector

logging.basicConfig(level=logging.INFO)

detector_port = GLOBALS.detector_port


@pytest.fixture()
async def sio() -> AsyncGenerator:
    sio_async_client = socketio.AsyncClient()
    try_connect = True
    count = 0
    while try_connect:
        count += 1
        try:
            await sio_async_client.connect(f"ws://localhost:{detector_port}", socketio_path="/ws/socket.io", wait_timeout=30)
            try_connect = False
        except Exception:
            logging.warning('trying again')
            await asyncio.sleep(1)
        if count == 5:
            raise Exception('Could not connect to sio')

    assert sio_async_client.transport() == 'websocket'
    yield sio_async_client
    await sio_async_client.disconnect()


@pytest.fixture()
async def test_detector_node():
    os.environ['LOOP_ORGANIZATION'] = 'zauberzeug'
    os.environ['LOOP_PROJECT'] = 'demo'

    model_info = ModelInformation(
        id='some_uuid', host='some_host', organization='zauberzeug', project='test', version='1',
        categories=[Category(id='some_id_1', name='some_category_name_1'),
                    Category(id='some_id_2', name='some_category_name_2'),
                    Category(id='some_id_3', name='some_category_name_3')])

    detector = MockDetector(model_format='mocked')
    node = DetectorNode(name='test', detector=detector)
    detector.model_info = model_info  # pylint: disable=protected-access
    await port_is(free=True)

    multiprocessing.set_start_method('fork', force=True)
    assert multiprocessing.get_start_method() == 'fork'
    proc = Process(target=uvicorn.run,
                   args=(node,),
                   kwargs={
                       "host": "127.0.0.1",
                       "port": detector_port,
                   }, daemon=True)
    proc.start()
    await port_is(free=False)
    yield node

    try:
        await node._on_shutdown()  # pylint: disable=protected-access
    except Exception:
        logging.exception('error while shutting down node')

    try:
        proc.kill()
    except Exception:  # for python 3.6
        proc.terminate()
    proc.join()


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


async def port_is(free: bool):
    for _ in range(10):
        if not free and is_port_in_use(detector_port):
            return
        if free and not is_port_in_use(detector_port):
            return
        await asyncio.sleep(0.5)
    raise Exception(f'port {detector_port} is {"not" if free else ""} free')


@pytest.fixture(autouse=True, scope='function')
def data_folder():
    GLOBALS.data_folder = '/tmp/learning_loop_lib_data'
    shutil.rmtree(GLOBALS.data_folder, ignore_errors=True)
    yield
    shutil.rmtree(GLOBALS.data_folder, ignore_errors=True)
