import asyncio
import re

import pytest

from ...data_classes import TrainerState
from .state_helper import assert_training_state, create_active_training_file
from .testing_trainer_logic import TestingTrainerLogic

# pylint: disable=protected-access


async def test_training_process_is_stopped_when_trainer_reports_error(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer
    create_active_training_file(trainer, training_state=TrainerState.TrainModelDownloaded)
    trainer._init_from_last_training()
    _ = asyncio.get_running_loop().create_task(trainer._run())

    await assert_training_state(trainer.training, TrainerState.TrainingRunning, timeout=1, interval=0.001)
    trainer.error_msg = 'some_error'
    await assert_training_state(trainer.training, TrainerState.TrainModelDownloaded, timeout=6, interval=0.001)


@pytest.mark.skip(reason='The since_last_start flag is deprecated.')
async def test_log_can_provide_only_data_for_current_run(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer
    create_active_training_file(trainer, training_state=TrainerState.TrainModelDownloaded)
    trainer._init_from_last_training()
    _ = asyncio.get_running_loop().create_task(trainer._run())

    await assert_training_state(trainer.training, TrainerState.TrainingRunning, timeout=1, interval=0.001)
    await asyncio.sleep(0.1)  # give tests a bit time to to check for the state

    assert trainer._executor is not None
    assert len(re.findall('Starting executor', str(trainer._executor.get_log_by_lines()))) == 1

    trainer.error_msg = 'some_error'
    await assert_training_state(trainer.training, TrainerState.TrainModelDownloaded, timeout=6, interval=0.001)
    trainer.error_msg = None
    await assert_training_state(trainer.training, TrainerState.TrainingRunning, timeout=1, interval=0.001)
    await asyncio.sleep(1)

    assert len(re.findall('Starting executor', str(trainer._executor.get_log_by_lines()))) > 1
    # Here only the current run is provided
    # assert len(re.findall('Starting executor', str(trainer._executor.get_log_by_lines(since_last_start=True)))) == 1
