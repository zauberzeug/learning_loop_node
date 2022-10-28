import multiprocessing
import pytest
from learning_loop_node import DetectorNode, ModelInformation
from learning_loop_node.detector import Outbox
from learning_loop_node.data_classes import Category
from testing_detector import TestingDetector
import uvicorn
from multiprocessing import Process, log_to_stderr
import logging
import icecream
import os
import socket
import asyncio
from typing import Generator
import socketio
from glob import glob

logging.basicConfig(level=logging.INFO)

# show ouptut from uvicorn server https://stackoverflow.com/a/66132186/364388
log_to_stderr(logging.INFO)

icecream.install()


def pytest_configure():
    pytest.detector_port = 5000


def should_have_segmentations(request) -> bool:
    should_have_segmentations = False
    try:
        should_have_segmentations = request.param
    except:
        pass
    return should_have_segmentations


@pytest.fixture()
async def test_detector_node(request):
    os.environ['ORGANIZATION'] = 'zauberzeug'
    os.environ['PROJECT'] = 'demo'

    model_info = ModelInformation(id='some_uuid', host='some_host', organization='zauberzeug',
                                  project='test', version='1', categories=[Category(id='some_id', name='some_category_name')])
    segmentations = should_have_segmentations(request)

    det = TestingDetector(segmentation_detections=segmentations)
    det.init(model_info=model_info, model_root_path='')
    node = DetectorNode(name='test', detector=det)
    await port_is(free=True)

    multiprocessing.set_start_method('fork', force=True)
    assert multiprocessing.get_start_method() == 'fork'
    proc = Process(target=uvicorn.run,
                   args=(node,),
                   kwargs={
                       "host": "127.0.0.1",
                       "port": pytest.detector_port,

                   }, daemon=True)
    proc.start()
    await port_is(free=False)
    yield node

    try:
        await node.shutdown()
    except:
        logging.exception('error while shutting down node')

    try:
        proc.kill()
    except:  # for python 3.6
        proc.terminate()
    proc.join()

# from https://stackoverflow.com/a/52872579/364388


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


async def port_is(free: bool):
    for i in range(10):
        if not free and is_port_in_use(pytest.detector_port):
            return
        if free and not is_port_in_use(pytest.detector_port):
            return
        else:
            await asyncio.sleep(0.5)
    raise Exception(f'port {pytest.detector_port} is {"not" if free else ""} free')


@pytest.fixture()
async def sio_client() -> Generator:
    sio = socketio.AsyncClient()
    try_connect = True
    retry_count = 0
    while try_connect:
        try:
            await sio.connect(f"ws://localhost:{pytest.detector_port}", socketio_path="/ws/socket.io")
            try_connect = False
        except:
            logging.warning('trying again')
            await asyncio.sleep(1)
        retry_count += 1
        if retry_count > 10:
            raise Exception('Max Retry')

    yield sio
    await sio.disconnect()


def get_outbox_files(outbox: Outbox):
    files = glob(f'{outbox.path}/**/*', recursive=True)
    return [file for file in files if os.path.isfile(file)]
