import time
from typing import Callable
from learning_loop_node.trainer.trainer import Trainer
from learning_loop_node.trainer.trainer_node import TrainerNode
from learning_loop_node.trainer.tests.testing_trainer import TestingTrainer
from learning_loop_node.trainer.training import Training
from uuid import uuid4
from learning_loop_node.context import Context
from learning_loop_node.trainer import active_training
import asyncio
import os
import pytest
import signal
import logging


@pytest.fixture
def remove_training_file(autouse=True):
    os.remove('last_training.json')
    yield
    os.remove('last_training.json')


def create_training() -> Training:
    context = Context(organization='zauberzeug', project='demo')
    training = Training(
        id=str(uuid4()),
        context=context,
        project_folder='',
        images_folder='')
    return training


def test_fixture_trainer_node(test_trainer_node):
    assert isinstance(test_trainer_node, TrainerNode)
    assert isinstance(test_trainer_node.trainer, TestingTrainer)


def test_save_load_training():
    training = create_training()
    training.training_state = 'preparing'
    active_training.save(training)
    training = active_training.load()
    assert training.training_state == 'preparing'


async def test_abort_preparing():
    trainer = Trainer(model_format='mocked')
    details = {'categories': [],
               'id': 'some_id',
               'training_number': 0,
               'resolution': 800,
               'flip_rl': False,
               'flip_ud': False}

    assert trainer.training is None
    trainer.init(Context(organization='zauberzeug', project='demo'), details)
    training_task = asyncio.get_running_loop().create_task(trainer.prepare())

    await asyncio.sleep(0.1)
    assert trainer.training is not None
    assert trainer.training.training_state == 'data_downloading'
    assert trainer.prepare_task is not None
    assert_training_file(exists=True)

    trainer.stop()
    await asyncio.sleep(0.0)

    assert trainer.prepare_task.cancelled() == True
    await asyncio.sleep(0.1)
    assert trainer.prepare_task is None
    assert trainer.training == None
    assert_training_file(exists=False)

    training_task.cancel()


async def test_abort_download_model():
    trainer = TestingTrainer()
    details = {'categories': [],
               'id': '00000000-1111-2222-3333-555555555555',
               'training_number': 0,
               'resolution': 800,
               'flip_rl': False,
               'flip_ud': False}

    assert trainer.training is None
    trainer.init(Context(organization='zauberzeug', project='demo'), details)
    await trainer.prepare()

    assert trainer.training.training_state == 'data_downloaded'
    assert_training_file(exists=True)

    download_task = asyncio.get_running_loop().create_task(trainer.download_model())

    await asyncio.sleep(0.0)

    assert trainer.download_model_task.cancelled() == False
    await asyncio.sleep(0.0)
    assert trainer.training.training_state == 'train_model_downloading'
    trainer.stop()

    await asyncio.sleep(0.0)

    assert trainer.download_model_task.cancelled() == True
    await asyncio.sleep(0.1)
    assert trainer.download_model_task is None
    assert trainer.training == None
    assert_training_file(exists=False)

    download_task.cancel()


async def test_initialized_training_can_be_resumed(test_trainer_node: TrainerNode):
    # Generate File on disc
    trainer = TestingTrainer()
    details = {'categories': [],
               'id': '00000000-1111-2222-3333-555555555555',
               'training_number': 0,
               'resolution': 800,
               'flip_rl': False,
               'flip_ud': False}
    trainer.init(Context(organization='zauberzeug', project='demo'), details)

    assert test_trainer_node.trainer.training is None
    train_task = asyncio.get_running_loop().create_task(test_trainer_node.train())
    await asyncio.sleep(0.0)
    await assert_training_state(test_trainer_node.trainer.training, 'initialized', timeout=1, interval=0.01)

    await assert_training_state(test_trainer_node.trainer.training, 'data_downloading', timeout=1, interval=0.001)
    await assert_training_state(test_trainer_node.trainer.training, 'data_downloaded', timeout=1, interval=0.001)


async def test_data_downloaded_training_can_be_resumed(test_trainer_node: TrainerNode):
    # Generate File on disc
    trainer = TestingTrainer()
    details = {'categories': [],
               'id': '00000000-1111-2222-3333-555555555555',
               'training_number': 0,
               'resolution': 800,
               'flip_rl': False,
               'flip_ud': False}
    trainer.init(Context(organization='zauberzeug', project='demo'), details)
    await trainer.prepare()

    assert test_trainer_node.trainer.training is None
    train_task = asyncio.get_running_loop().create_task(test_trainer_node.train())
    await asyncio.sleep(0.0)

    # await assert_training_state(test_trainer_node.trainer.training, 'train_model_downloading', timeout=1, interval=0.01)
    await assert_training_state(test_trainer_node.trainer.training, 'train_model_downloaded', timeout=1, interval=0.01)


async def test_stop_running_training():
    # Generate File on disc
    trainer = TestingTrainer()
    details = {'categories': [],
               'id': '00000000-1111-2222-3333-555555555555',
               'training_number': 0,
               'resolution': 800,
               'flip_rl': False,
               'flip_ud': False}
    trainer.init(Context(organization='zauberzeug', project='demo'), details)
    await trainer.prepare()
    await trainer.download_model()
    assert trainer.training.training_state == 'train_model_downloaded'
    assert trainer.executor is None

    train_task = asyncio.get_running_loop().create_task(trainer.run_training())
    await assert_training_state(trainer.training, 'training_running', timeout=1, interval=0.001)
    await condition(lambda: trainer.executor is not None, timeout=1, interval=0.01)

    trainer.stop()

    await assert_training_state(trainer.training, 'training_finished', timeout=1, interval=0.001)

    assert trainer.training is not None
    assert_training_file(exists=True)


def assert_training_file(exists: bool) -> None:
    assert active_training.exists() == exists


async def assert_training_state(training: Training, state: str, timeout: float, interval: float) -> None:
    try:
        await condition(lambda: training.training_state == state, timeout=timeout, interval=interval)
    except TimeoutError:
        logging.error('h############################')
        msg = f"Trainer state should be '{state}' after {timeout} seconds, but is {training.training_state}"
        raise AssertionError(msg)
    except Exception as e:
        logging.exception('##### was ist das hier?')
        raise


class Timeout:
    # see https://stackoverflow.com/a/22348885/4082686
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


async def condition(condition: Callable, *, timeout: float = 1.0, interval: float = 0.1):
    start = time.time()
    while not condition():
        if time.time() > start + timeout:
            raise TimeoutError(f'condition {condition} took longer than {timeout}s')
        await asyncio.sleep(interval)
