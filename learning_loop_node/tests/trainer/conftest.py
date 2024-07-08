from ...globals import GLOBALS
import shutil
import logging
import os
import socket
from multiprocessing import log_to_stderr

import icecream
import pytest

from ...data_classes import Context
from ...trainer.trainer_node import TrainerNode
from .testing_trainer_logic import TestingTrainerLogic

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
                                        'id': '00000000-0000-0000-0000-000000000012',  # version 1.2 of demo project
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
                                        'id': '00000000-0000-0000-0000-000000000012',  # version 1.2 of demo project
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
