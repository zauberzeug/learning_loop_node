import asyncio

from ....data_classes import TrainerState
from ...test_helper import condition
from ..state_helper import assert_training_state, create_active_training_file
from ..testing_trainer_logic import TestingTrainerLogic

# pylint: disable=protected-access


async def test_successful_training(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer

    create_active_training_file(trainer, training_state=TrainerState.TrainModelDownloaded)
    trainer._init_from_last_training()

    _ = asyncio.get_running_loop().create_task(trainer._run())

    await condition(lambda: trainer._executor and trainer._executor.is_running(), timeout=1, interval=0.01)
    await assert_training_state(trainer.training, TrainerState.TrainingRunning, timeout=1, interval=0.01)
    assert trainer.start_training_task is not None

    assert trainer._executor is not None
    await trainer.stop()  # NOTE normally a training terminates itself
    await assert_training_state(trainer.training, TrainerState.TrainingFinished, timeout=1, interval=0.001)

    assert trainer.training.training_state == TrainerState.TrainingFinished
    assert trainer.node.last_training_io.load() == trainer.training


async def test_stop_running_training(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer

    create_active_training_file(trainer, training_state=TrainerState.TrainModelDownloaded)
    trainer._init_from_last_training()

    _ = asyncio.get_running_loop().create_task(trainer._run())

    await condition(lambda: trainer._executor and trainer._executor.is_running(), timeout=1, interval=0.01)
    await assert_training_state(trainer.training, TrainerState.TrainingRunning, timeout=1, interval=0.01)
    assert trainer.start_training_task is not None

    await trainer.stop()
    await assert_training_state(trainer.training, TrainerState.TrainingFinished, timeout=2, interval=0.01)

    assert trainer.training.training_state == TrainerState.TrainingFinished
    assert trainer.node.last_training_io.load() == trainer.training


async def test_training_can_maybe_resumed(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer

    # NOTE e.g. when a node-computer is restarted
    create_active_training_file(trainer, training_state=TrainerState.TrainModelDownloaded)
    trainer._init_from_last_training()
    trainer._can_resume_flag = True

    _ = asyncio.get_running_loop().create_task(trainer._run())

    await condition(lambda: trainer._executor and trainer._executor.is_running(), timeout=1, interval=0.01)
    await assert_training_state(trainer.training, TrainerState.TrainingRunning, timeout=1, interval=0.001)
    assert trainer.start_training_task is not None

    assert trainer._executor is not None
    await trainer._executor.stop_and_wait()  # NOTE normally a training terminates itself e.g
    await assert_training_state(trainer.training, TrainerState.TrainingFinished, timeout=1, interval=0.001)

    assert trainer.training.training_state == TrainerState.TrainingFinished
    assert trainer.node.last_training_io.load() == trainer.training
