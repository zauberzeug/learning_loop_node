
from uuid import uuid4

from ...data_classes import Context, TrainerState, Training
from ...trainer.io_helpers import LastTrainingIO
from ...trainer.trainer_node import TrainerNode
from .testing_trainer_logic import TestingTrainerLogic


def create_training() -> Training:
    context = Context(organization='zauberzeug', project='demo')
    training = Training(
        id=str(uuid4()),
        context=context,
        project_folder='',
        images_folder='',
        training_folder='')
    return training


def test_fixture_trainer_node(test_initialized_trainer_node):
    assert isinstance(test_initialized_trainer_node, TrainerNode)
    assert isinstance(test_initialized_trainer_node.trainer_logic, TestingTrainerLogic)


def test_save_load_training():
    training = create_training()
    last_training_io = LastTrainingIO('00000000-0000-0000-0000-000000000000')
    training.training_state = TrainerState.Preparing
    last_training_io.save(training)

    training = last_training_io.load()
    assert training.training_state == TrainerState.Preparing
