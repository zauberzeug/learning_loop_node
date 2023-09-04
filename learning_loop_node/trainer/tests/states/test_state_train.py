import asyncio

from learning_loop_node.tests.test_helper import condition
from learning_loop_node.trainer.tests.state_helper import (
    assert_training_state, create_active_training_file)
from learning_loop_node.trainer.tests.testing_trainer import TestingTrainer


async def test_successful_training(test_initialized_trainer: TestingTrainer):
    trainer = test_initialized_trainer

    create_active_training_file(trainer, training_state='train_model_downloaded')
    trainer.load_active_training()

    _ = asyncio.get_running_loop().create_task(trainer.train())

    await assert_training_state(trainer.training, 'training_running', timeout=1, interval=0.001)
    assert trainer.train_task is not None
    assert trainer.train_task.__name__ == 'start_training'

    assert trainer.executor is not None
    trainer.executor.stop()  # NOTE normally a training terminates itself
    await assert_training_state(trainer.training, 'training_finished', timeout=1, interval=0.001)

    assert trainer.training.training_state == 'training_finished'
    assert trainer.last_training_io.load() == trainer.training


async def test_stop_running_training(test_initialized_trainer: TestingTrainer):
    trainer = test_initialized_trainer

    create_active_training_file(trainer, training_state='train_model_downloaded')
    trainer.load_active_training()

    _ = asyncio.get_running_loop().create_task(trainer.train())

    await condition(lambda: trainer.executor and trainer.executor.is_process_running(), timeout=1, interval=0.01)
    await assert_training_state(trainer.training, 'training_running', timeout=1, interval=0.001)
    assert trainer.train_task is not None
    assert trainer.train_task.__name__ == 'start_training'

    trainer.stop()
    await assert_training_state(trainer.training, 'training_finished', timeout=1, interval=0.001)

    assert trainer.training.training_state == 'training_finished'
    assert trainer.last_training_io.load() == trainer.training


async def test_training_can_maybe_resumed(test_initialized_trainer: TestingTrainer):
    trainer = test_initialized_trainer

    # NOTE e.g. when a node-computer is restarted
    create_active_training_file(trainer, training_state='train_model_downloaded')
    trainer.load_active_training()
    trainer._can_resume = True  # pylint: disable=protected-access

    _ = asyncio.get_running_loop().create_task(trainer.train())

    await condition(lambda: trainer.executor and trainer.executor.is_process_running(), timeout=1, interval=0.01)
    await assert_training_state(trainer.training, 'training_running', timeout=1, interval=0.001)
    assert trainer.train_task is not None
    assert trainer.train_task.__name__ == 'resume'

    assert trainer.executor is not None
    trainer.executor.stop()  # NOTE normally a training terminates itself e.g
    await assert_training_state(trainer.training, 'training_finished', timeout=1, interval=0.001)

    assert trainer.training.training_state == 'training_finished'
    assert trainer.last_training_io.load() == trainer.training
