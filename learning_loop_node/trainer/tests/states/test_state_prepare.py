import asyncio

from learning_loop_node.data_classes import Context
from learning_loop_node.trainer.tests.state_helper import (
    assert_training_state, create_active_training_file)
from learning_loop_node.trainer.tests.testing_trainer_logic import \
    TestingTrainerLogic
from learning_loop_node.trainer.trainer_logic import TrainerLogic

error_key = 'prepare'


def trainer_has_error(trainer: TrainerLogic):
    return trainer.errors.has_error_for(error_key)


async def test_preparing_is_successful(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer
    create_active_training_file(trainer)
    trainer.load_last_training()

    await trainer.prepare()
    assert trainer_has_error(trainer) is False
    assert trainer.training.training_state == 'data_downloaded'
    assert trainer.training.data is not None
    assert trainer.node.last_training_io.load() == trainer.training


async def test_abort_preparing(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer
    create_active_training_file(trainer)
    trainer.load_last_training()

    _ = asyncio.get_running_loop().create_task(trainer.run())
    await assert_training_state(trainer.training, 'data_downloading', timeout=1, interval=0.001)

    await trainer.stop()
    await asyncio.sleep(0.1)

    assert trainer._training is None  # pylint: disable=protected-access
    assert trainer.node.last_training_io.exists() is False


async def test_request_error(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer
    create_active_training_file(trainer, context=Context(
        organization='zauberzeug', project='some_bad_project'))
    trainer.load_last_training()

    _ = asyncio.get_running_loop().create_task(trainer.run())
    await assert_training_state(trainer.training, 'data_downloading', timeout=3, interval=0.001)
    await assert_training_state(trainer.training, 'initialized', timeout=3, interval=0.001)

    assert trainer_has_error(trainer)
    assert trainer._training is not None  # pylint: disable=protected-access
    assert trainer.training.training_state == 'initialized'
    assert trainer.node.last_training_io.load() == trainer.training
