import asyncio
import re

from learning_loop_node.trainer.tests.states.state_helper import (
    assert_training_state, create_active_training_file)
from learning_loop_node.trainer.tests.testing_trainer import TestingTrainer
from learning_loop_node.trainer.trainer_node import TrainerNode


async def test_training_process_is_stopped_when_trainer_reports_error(test_initialized_trainer_node: TrainerNode):
    create_active_training_file(training_state='train_model_downloaded')
    test_initialized_trainer_node.trainer = TestingTrainer()
    trainer = test_initialized_trainer_node.trainer

    assert trainer._training is not None and trainer._active_training_io is not None
    trainer._training = trainer._active_training_io.load()  # normally done by node

    _ = asyncio.get_running_loop().create_task(trainer.train(
        uuid=test_initialized_trainer_node.uuid, sio_client=test_initialized_trainer_node._sio_client))

    await assert_training_state(trainer._training, 'training_running', timeout=1, interval=0.001)

    trainer.error_msg = 'some_error'
    await assert_training_state(trainer._training, 'train_model_downloaded', timeout=6, interval=0.001)


async def test_log_can_provide_only_data_for_current_run(test_initialized_trainer_node: TrainerNode):
    create_active_training_file(training_state='train_model_downloaded')
    test_initialized_trainer_node.trainer = TestingTrainer()
    trainer = test_initialized_trainer_node.trainer

    assert trainer._training is not None and trainer._active_training_io is not None
    trainer._training = trainer._active_training_io.load()  # normally done by node

    _ = asyncio.get_running_loop().create_task(trainer.train(
        uuid=test_initialized_trainer_node.uuid, sio_client=test_initialized_trainer_node._sio_client))
    await assert_training_state(trainer._training, 'training_running', timeout=1, interval=0.001)

    assert len(re.findall('Starting executor', str(trainer.executor.get_log_by_lines()))) == 1

    trainer.error_msg = 'some_error'
    await assert_training_state(trainer._training, 'train_model_downloaded', timeout=6, interval=0.001)
    trainer.error_msg = None
    await assert_training_state(trainer._training, 'training_running', timeout=1, interval=0.001)
    await asyncio.sleep(1)

    assert len(re.findall('Starting executor', str(trainer.executor.get_log_by_lines()))) > 1
    # Here only the current run is provided
    assert len(re.findall('Starting executor', str(trainer.executor.get_log_by_lines(since_last_start=True)))) == 1
