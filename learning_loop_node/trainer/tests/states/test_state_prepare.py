import asyncio

from learning_loop_node.data_classes import Context
from learning_loop_node.trainer.tests.states.state_helper import (
    assert_training_state, create_active_training_file)
from learning_loop_node.trainer.tests.testing_trainer import TestingTrainer
from learning_loop_node.trainer.trainer import Trainer

error_key = 'prepare'


def trainer_has_error(trainer: Trainer):
    return trainer.errors.has_error_for(error_key)


async def test_preparing_is_successful(test_initialized_trainer: TestingTrainer):
    trainer = test_initialized_trainer
    assert trainer._training is not None and trainer._active_training_io is not None

    create_active_training_file()
    trainer.load_active_training()

    await trainer.prepare()

    assert trainer_has_error(trainer) is False
    assert trainer._training.training_state == 'data_downloaded'
    assert trainer._training.data is not None
    assert trainer._active_training_io.load() == trainer._training


async def test_abort_preparing(test_initialized_trainer: TestingTrainer):
    trainer = test_initialized_trainer
    assert trainer._training is not None and trainer._active_training_io is not None

    create_active_training_file()
    trainer.load_active_training()

    _ = asyncio.get_running_loop().create_task(trainer.train(None, None))
    await assert_training_state(trainer._training, 'data_downloading', timeout=1, interval=0.001)

    trainer.stop()
    await asyncio.sleep(0.1)

    assert trainer._training is None
    assert trainer._active_training_io.exists() is False


async def test_request_error(test_initialized_trainer: TestingTrainer):
    trainer = test_initialized_trainer
    assert trainer._training is not None and trainer._active_training_io is not None

    create_active_training_file(context=Context(
        organization='zauberzeug', project='some_bad_project'))
    trainer.load_active_training()

    _ = asyncio.get_running_loop().create_task(trainer.train(None, None))
    await assert_training_state(trainer._training, 'data_downloading', timeout=3, interval=0.001)
    await assert_training_state(trainer._training, 'initialized', timeout=3, interval=0.001)

    assert trainer_has_error(trainer)
    assert trainer._training is not None
    assert trainer._training.training_state == 'initialized'
    assert trainer._active_training_io.load() == trainer._training
