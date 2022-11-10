from learning_loop_node.trainer.trainer_node import TrainerNode
from learning_loop_node.trainer.tests.testing_trainer import TestingTrainer
from learning_loop_node.trainer.training import Training
from uuid import uuid4
from learning_loop_node.context import Context
from learning_loop_node.trainer import active_training
import os
import pytest
import signal


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
