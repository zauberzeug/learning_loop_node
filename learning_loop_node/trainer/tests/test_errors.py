import asyncio
import re

from learning_loop_node.trainer.tests.state_helper import (
    assert_training_state, create_active_training_file)
from learning_loop_node.trainer.tests.testing_trainer_logic import \
    TestingTrainerLogic


async def test_training_process_is_stopped_when_trainer_reports_error(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer
    create_active_training_file(trainer, training_state='train_model_downloaded')
    trainer.load_last_training()
    _ = asyncio.get_running_loop().create_task(trainer.run())

    await assert_training_state(trainer.training, 'training_running', timeout=1, interval=0.001)
    trainer.error_msg = 'some_error'
    await assert_training_state(trainer.training, 'train_model_downloaded', timeout=6, interval=0.001)


async def test_log_can_provide_only_data_for_current_run(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer
    create_active_training_file(trainer, training_state='train_model_downloaded')
    trainer.load_last_training()
    _ = asyncio.get_running_loop().create_task(trainer.run())

    await assert_training_state(trainer.training, 'training_running', timeout=1, interval=0.001)
    assert trainer._executor is not None
    assert len(re.findall('Starting executor', str(trainer._executor.get_log_by_lines()))) == 1

    trainer.error_msg = 'some_error'
    await assert_training_state(trainer.training, 'train_model_downloaded', timeout=6, interval=0.001)
    trainer.error_msg = None
    await assert_training_state(trainer.training, 'training_running', timeout=1, interval=0.001)
    await asyncio.sleep(1)

    assert len(re.findall('Starting executor', str(trainer._executor.get_log_by_lines()))) > 1
    # Here only the current run is provided
    assert len(re.findall('Starting executor', str(trainer._executor.get_log_by_lines(since_last_start=True)))) == 1
