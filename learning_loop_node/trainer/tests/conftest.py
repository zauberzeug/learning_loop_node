import logging
import os
import socket
from multiprocessing import log_to_stderr

import icecream
import pytest

from learning_loop_node.data_classes import Context
from learning_loop_node.trainer.tests.testing_trainer_logic import TestingTrainerLogic
from learning_loop_node.trainer.trainer_node import TrainerNode

# pylint: disable=protected-access

logging.basicConfig(level=logging.INFO)
# show ouptut from uvicorn server https://stackoverflow.com/a/66132186/364388
log_to_stderr(logging.INFO)

icecream.install()


@pytest.fixture()
async def test_initialized_trainer_node():
    os.environ['ORGANIZATION'] = 'zauberzeug'
    os.environ['PROJECT'] = 'demo'

    trainer = TestingTrainerLogic()
    node = TrainerNode(name='test', trainer_logic=trainer, uuid='NOD30000-0000-0000-0000-000000000000')
    trainer._node = node
    trainer._init_new_training(context=Context(organization='zauberzeug', project='demo'),
                               details={'categories': [],
                                        'id': '917d5c7f-403d-7e92-f95f-577f79c2273a',  # version 1.2 of demo project
                                        'training_number': 0,
                                        'resolution': 800,
                                        'flip_rl': False,
                                        'flip_ud': False})
    await node._on_startup()
    yield node
    await node._on_shutdown()


@pytest.fixture()
async def test_initialized_trainer():

    trainer = TestingTrainerLogic()
    node = TrainerNode(name='test', trainer_logic=trainer, uuid='NODE-000-0000-0000-0000-000000000000')

    await node._on_startup()
    trainer._node = node
    trainer._init_new_training(context=Context(organization='zauberzeug', project='demo'),
                               details={'categories': [],
                                        'id': '917d5c7f-403d-7e92-f95f-577f79c2273a',  # version 1.2 of demo project
                                        'training_number': 0,
                                        'resolution': 800,
                                        'flip_rl': False,
                                        'flip_ud': False})
    yield trainer
    try:
        await node._on_shutdown()
    except Exception:
        logging.exception('error while shutting down node')


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0
