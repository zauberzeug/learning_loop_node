if True:
    import logging
    logging.basicConfig(level=logging.INFO)
import multiprocessing
from shutil import ExecError
from learning_loop_node.trainer.tests.testing_trainer import TestingTrainer
from learning_loop_node.trainer.trainer_node import TrainerNode
import pytest

import uvicorn
from multiprocessing import Process, log_to_stderr
import icecream
import os
import socket
import asyncio
from glob import glob


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
    await node.startup()

    yield node

    try:
        await node.shutdown()
    except:
        logging.exception('error while shutting down node')


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


@pytest.fixture(autouse=True, scope='session')
def initialize_active_training():
    from learning_loop_node.trainer import active_training
    active_training.init('00000000-0000-0000-0000-000000000000')
    yield
