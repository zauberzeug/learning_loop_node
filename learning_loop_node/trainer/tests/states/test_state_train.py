import asyncio

from learning_loop_node.tests.test_helper import condition
from learning_loop_node.trainer.tests.state_helper import (
    assert_training_state, create_active_training_file)
from learning_loop_node.trainer.tests.testing_trainer_logic import \
    TestingTrainerLogic


async def test_successful_training(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer

    create_active_training_file(trainer, training_state='train_model_downloaded')
    trainer.load_last_training()

    _ = asyncio.get_running_loop().create_task(trainer.run())

    await assert_training_state(trainer.training, 'training_running', timeout=1, interval=0.001)
    assert trainer.start_training_task is not None
    assert trainer.start_training_task.__name__ == 'start_training'

    # pylint: disable=protected-access
    assert trainer._executor is not None
    trainer._executor.stop()  # NOTE normally a training terminates itself
    await assert_training_state(trainer.training, 'training_finished', timeout=1, interval=0.001)

    assert trainer.training.training_state == 'training_finished'
    assert trainer.node.last_training_io.load() == trainer.training


async def test_stop_running_training(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer

    create_active_training_file(trainer, training_state='train_model_downloaded')
    trainer.load_last_training()

    _ = asyncio.get_running_loop().create_task(trainer.run())

    await condition(lambda: trainer._executor and trainer._executor.is_process_running(), timeout=1, interval=0.01)  # pylint: disable=protected-access
    await assert_training_state(trainer.training, 'training_running', timeout=1, interval=0.001)
    assert trainer.start_training_task is not None
    assert trainer.start_training_task.__name__ == 'start_training'

    await trainer.stop()
    await assert_training_state(trainer.training, 'training_finished', timeout=1, interval=0.001)

    assert trainer.training.training_state == 'training_finished'
    assert trainer.node.last_training_io.load() == trainer.training


async def test_training_can_maybe_resumed(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer

    # NOTE e.g. when a node-computer is restarted
    create_active_training_file(trainer, training_state='train_model_downloaded')
    trainer.load_last_training()
    trainer._can_resume = True  # pylint: disable=protected-access

    _ = asyncio.get_running_loop().create_task(trainer.run())

    await condition(lambda: trainer._executor and trainer._executor.is_process_running(), timeout=1, interval=0.01)  # pylint: disable=protected-access
    await assert_training_state(trainer.training, 'training_running', timeout=1, interval=0.001)
    assert trainer.start_training_task is not None
    assert trainer.start_training_task.__name__ == 'resume'

    # pylint: disable=protected-access
    assert trainer._executor is not None
    trainer._executor.stop()  # NOTE normally a training terminates itself e.g
    await assert_training_state(trainer.training, 'training_finished', timeout=1, interval=0.001)

    assert trainer.training.training_state == 'training_finished'
    assert trainer.node.last_training_io.load() == trainer.training
