import multiprocessing
from shutil import ExecError
from learning_loop_node.trainer.tests.testing_trainer import TestingTrainer
from learning_loop_node.trainer.trainer_node import TrainerNode
import pytest
from learning_loop_node import DetectorNode, ModelInformation
from learning_loop_node.detector import Outbox
from learning_loop_node.data_classes import Category

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
    pytest.trainer_port = 5001


@pytest.fixture()
async def test_trainer_node(request):
    os.environ['ORGANIZATION'] = 'zauberzeug'
    os.environ['PROJECT'] = 'demo'

    trainer = TestingTrainer()

    node = TrainerNode(name='test', trainer=trainer, uuid='00000000-0000-0000-0000-000000000000')
    await port_is(free=True)

    multiprocessing.set_start_method('fork', force=True)
    assert multiprocessing.get_start_method() == 'fork'
    proc = Process(target=uvicorn.run,
                   args=(node,),
                   kwargs={
                       "host": "127.0.0.1",
                       "port": pytest.trainer_port,

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


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


async def port_is(free: bool):
    for i in range(10):
        if not free and is_port_in_use(pytest.trainer_port):
            return
        if free and not is_port_in_use(pytest.trainer_port):
            return
        else:
            await asyncio.sleep(0.5)
    raise Exception(f'port {pytest.trainer_port} is {"not" if free else ""} free')


@pytest.fixture(autouse=True, scope='session')
def initialize_active_training():
    from learning_loop_node.trainer import active_training
    active_training.init('00000000-0000-0000-0000-000000000000')
    yield
