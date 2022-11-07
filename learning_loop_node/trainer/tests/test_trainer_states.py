from learning_loop_node.trainer.trainer import Trainer
from learning_loop_node.trainer.trainer_node import TrainerNode
from learning_loop_node.trainer.tests.testing_trainer import TestingTrainer
from learning_loop_node.trainer.training import Training
from uuid import uuid4
from learning_loop_node.context import Context
from learning_loop_node.trainer import training as training_module
import asyncio


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
    training_module.save(training)
    training = training_module.load()
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
    training_task = asyncio.get_running_loop().create_task(
        trainer.begin_training(Context(organization='zauberzeug', project='demo'), details))
    await asyncio.sleep(0.1)
    assert trainer.training is not None
    assert trainer.training.training_state == 'init'
    assert trainer.prepare_task is not None

    trainer.stop()
    await asyncio.sleep(0.0)

    assert trainer.prepare_task.cancelled() == True
    await asyncio.sleep(0.1)
    assert trainer.prepare_task is None
    assert trainer.training == None
    training_task.cancel()
