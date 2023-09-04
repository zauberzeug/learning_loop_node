
from uuid import uuid4

from learning_loop_node.data_classes import Context, Training, TrainingState
from learning_loop_node.trainer.tests.testing_trainer import TestingTrainer
from learning_loop_node.trainer.trainer_node import TrainerNode
from learning_loop_node.trainer.training_io_helpers import ActiveTrainingIO


def create_training() -> Training:
    context = Context(organization='zauberzeug', project='demo')
    training = Training(
        uuid=str(uuid4()),
        context=context,
        project_folder='',
        images_folder='',
        training_folder='')
    return training


def test_fixture_trainer_node(test_initialized_trainer_node):
    assert isinstance(test_initialized_trainer_node, TrainerNode)
    assert isinstance(test_initialized_trainer_node.trainer, TestingTrainer)


def test_save_load_training():
    training = create_training()
    active_training_io = ActiveTrainingIO.create_mocked_training_io()
    training.training_state = TrainingState.Preparing
    active_training_io.save(training)

    training = active_training_io.load()
    assert training.training_state == 'preparing'
