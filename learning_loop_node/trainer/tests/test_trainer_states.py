from learning_loop_node.tests.test_helper import condition
from learning_loop_node.trainer.tests.states.state_helper import assert_training_file, assert_training_state
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


async def test_model_downloaded_training_can_be_resumed(test_trainer_node: TrainerNode):
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

    assert test_trainer_node.trainer.training is None
    train_task = asyncio.get_running_loop().create_task(test_trainer_node.train())
    await asyncio.sleep(0.0)

    # await assert_training_state(test_trainer_node.trainer.training, 'train_model_downloading', timeout=1, interval=0.01)
    await assert_training_state(test_trainer_node.trainer.training, 'training_running', timeout=1, interval=0.01)


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
