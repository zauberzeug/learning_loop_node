import shutil
import asyncio
import logging
import multiprocessing
import os
import socket
from glob import glob
from multiprocessing import Process, log_to_stderr
from typing import AsyncGenerator

import pytest
import socketio
import uvicorn

from ...detector.detector_node import DetectorNode
from ...detector.outbox import Outbox
from ...globals import GLOBALS
from .testing_detector import TestingDetectorLogic

logging.basicConfig(level=logging.INFO)

# show ouptut from uvicorn server https://stackoverflow.com/a/66132186/364388
log_to_stderr(logging.INFO)

detector_port = GLOBALS.detector_port


def should_have_segmentations(request) -> bool:
    should_have_seg = False
    try:
        should_have_seg = request.param
    except Exception:
        pass
    return should_have_seg


@pytest.fixture()
async def test_detector_node():
    """Initializes and runs a detector testnode. Note that the running instance and the one the function returns are not the same instances!"""

    os.environ['ORGANIZATION'] = 'zauberzeug'
    os.environ['PROJECT'] = 'demo'

    detector = TestingDetectorLogic()
    node = DetectorNode(name='test', detector=detector)
    await port_is(free=True)

    multiprocessing.set_start_method('fork', force=True)
    assert multiprocessing.get_start_method() == 'fork'
    # print('\n\n',multiprocessing.get_start_method())

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

# from https://stackoverflow.com/a/52872579/364388


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


@pytest.fixture()
async def sio_client() -> AsyncGenerator[socketio.AsyncClient, None]:
    sio = socketio.AsyncClient()
    try_connect = True
    retry_count = 0
    while try_connect:
        try:
            await sio.connect(f"ws://localhost:{detector_port}", socketio_path="/ws/socket.io")
            try_connect = False
        except Exception as e:
            logging.warning(f"Connection failed with error: {str(e)}")
            logging.warning('trying again')
            await asyncio.sleep(5)
        retry_count += 1
        if retry_count > 10:
            raise Exception('Max Retry')

    yield sio
    await sio.disconnect()


def get_outbox_files(outbox: Outbox):
    files = glob(f'{outbox.path}/**/*', recursive=True)
    return [file for file in files if os.path.isfile(file)]

# ====================================== REDUNDANT FIXTURES IN ALL CONFTESTS ! ======================================


@pytest.fixture(autouse=True, scope='session')
def clear_loggers():
    """Remove handlers from all loggers"""
    # see https://github.com/pytest-dev/pytest/issues/5502
    yield

    loggers = [logging.getLogger()] + list(logging.Logger.manager.loggerDict.values())
    for logger in loggers:
        if not isinstance(logger, logging.Logger):
            continue
        handlers = getattr(logger, 'handlers', [])
        for handler in handlers:
            logger.removeHandler(handler)


@pytest.fixture(autouse=True, scope='function')
def data_folder():
    GLOBALS.data_folder = '/tmp/learning_loop_lib_data'
    shutil.rmtree(GLOBALS.data_folder, ignore_errors=True)
    os.makedirs(GLOBALS.data_folder, exist_ok=True)
    yield
    shutil.rmtree(GLOBALS.data_folder, ignore_errors=True)
