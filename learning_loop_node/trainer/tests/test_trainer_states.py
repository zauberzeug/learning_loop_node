import os
import signal
from uuid import uuid4

import pytest

from learning_loop_node.data_classes.context import Context
from learning_loop_node.trainer import active_training
from learning_loop_node.trainer.tests.testing_trainer import TestingTrainer
from learning_loop_node.trainer.trainer_node import TrainerNode
from learning_loop_node.trainer.training import Training


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
