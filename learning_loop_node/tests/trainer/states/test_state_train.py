from pytest_mock import MockerFixture

from ....enums import TrainerState
from ...test_helper import condition
from ..state_helper import assert_training_state, create_active_training_file
from ..testing_trainer_logic import TestingTrainerLogic

# pylint: disable=protected-access


async def test_successful_training(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer

    create_active_training_file(trainer, training_state=TrainerState.TrainModelDownloaded)
    trainer._init_from_last_training()

    trainer._begin_training_task()

    await condition(lambda: trainer._executor and trainer._executor.is_running(), timeout=1, interval=0.01)
    await assert_training_state(trainer.training, TrainerState.TrainingRunning, timeout=10, interval=0.01)
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

    trainer._begin_training_task()

    await condition(lambda: trainer._executor and trainer._executor.is_running(), timeout=1, interval=0.01)
    await assert_training_state(trainer.training, TrainerState.TrainingRunning, timeout=10, interval=0.01)
    assert trainer.start_training_task is not None

    await trainer.stop()
    await assert_training_state(trainer.training, TrainerState.TrainingFinished, timeout=2, interval=0.01)

    assert trainer.node.last_training_io.load() == trainer.training


async def test_stop_during_training_uploads_model(mocker: MockerFixture, test_initialized_trainer: TestingTrainerLogic):
    """stop() during TrainingRunning should let the wind-down states run,
    so the model ends up uploaded (model_uuid_for_detecting set)."""

    trainer = test_initialized_trainer
    mocker.patch('learning_loop_node.data_exchanger.DataExchanger.upload_model_get_uuid',
                 return_value='12345678-abcd-1234-abcd-123456789abc')

    create_active_training_file(trainer, training_state=TrainerState.TrainModelDownloaded)
    trainer._init_from_last_training()
    trainer._begin_training_task()

    await condition(lambda: trainer._executor and trainer._executor.is_running(), timeout=1, interval=0.01)
    await assert_training_state(trainer.training, TrainerState.TrainingRunning, timeout=10, interval=0.01)

    await trainer.stop()

    await condition(lambda: trainer._training is not None
                    and trainer.training.model_uuid_for_detecting == '12345678-abcd-1234-abcd-123456789abc',
                    timeout=10, interval=0.05)


async def test_abort_during_training(mocker: MockerFixture, test_initialized_trainer: TestingTrainerLogic):
    """abort() during TrainingRunning skips the wind-down states: no upload call
    is made, the training is cleared, and the on-disk last_training_io is removed."""
    trainer = test_initialized_trainer
    upload_mock = mocker.patch('learning_loop_node.data_exchanger.DataExchanger.upload_model_get_uuid',
                               return_value='12345678-abcd-1234-abcd-123456789abc')

    create_active_training_file(trainer, training_state=TrainerState.TrainModelDownloaded)
    trainer._init_from_last_training()
    trainer._begin_training_task()

    await condition(lambda: trainer._executor and trainer._executor.is_running(), timeout=1, interval=0.01)
    await assert_training_state(trainer.training, TrainerState.TrainingRunning, timeout=10, interval=0.01)

    await trainer.abort()
    await condition(lambda: trainer._training is None, timeout=1, interval=0.01)

    upload_mock.assert_not_called()
    assert trainer.node.last_training_io.exists() is False


async def test_training_can_maybe_resumed(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer

    # NOTE e.g. when a node-computer is restarted
    create_active_training_file(trainer, training_state=TrainerState.TrainModelDownloaded)
    trainer._init_from_last_training()
    trainer._can_resume_flag = True

    trainer._begin_training_task()

    await condition(lambda: trainer._executor and trainer._executor.is_running(), timeout=1, interval=0.01)
    await assert_training_state(trainer.training, TrainerState.TrainingRunning, timeout=10, interval=0.001)
    assert trainer.start_training_task is not None

    assert trainer._executor is not None
    await trainer._executor.stop_and_wait()  # NOTE normally a training terminates itself e.g
    await assert_training_state(trainer.training, TrainerState.TrainingFinished, timeout=1, interval=0.001)

    assert trainer.node.last_training_io.load() == trainer.training
