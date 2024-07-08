import asyncio

from pytest_mock import MockerFixture

from ....data_classes import Context, TrainerState
from ....trainer.trainer_logic import TrainerLogic
from ..state_helper import assert_training_state, create_active_training_file
from ..testing_trainer_logic import TestingTrainerLogic

# pylint: disable=protected-access
error_key = 'upload_model'


def trainer_has_error(trainer: TrainerLogic):
    return trainer.errors.has_error_for(error_key)


async def test_successful_upload(mocker: MockerFixture, test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer
    mock_upload_model_for_training(mocker, 'new_model_id')

    create_active_training_file(trainer)
    trainer._init_from_last_training()

    train_task = asyncio.get_running_loop().create_task(
        trainer._perform_state('upload_model', TrainerState.TrainModelUploading, TrainerState.TrainModelUploaded, trainer._upload_model))

    await assert_training_state(trainer.training, TrainerState.TrainModelUploading, timeout=1, interval=0.001)
    await train_task

    assert trainer_has_error(trainer) is False
    assert trainer.training.training_state == TrainerState.TrainModelUploaded
    assert trainer.training.model_uuid_for_detecting is not None
    assert trainer.node.last_training_io.load() == trainer.training


async def test_abort_upload_model(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer

    create_active_training_file(trainer, training_state=TrainerState.ConfusionMatrixSynced)
    trainer._init_from_last_training()

    _ = asyncio.get_running_loop().create_task(trainer._run())

    await assert_training_state(trainer.training, TrainerState.TrainModelUploading, timeout=1, interval=0.001)

    await trainer.stop()
    await asyncio.sleep(0.1)

    assert trainer._training is None  # pylint: disable=protected-access
    assert trainer.node.last_training_io.exists() is False


async def test_bad_server_response_content(test_initialized_trainer: TestingTrainerLogic):
    """Set the training state to confusion_matrix_synced and try to upload the model.
    This should fail because the server response is not a valid model id.
    The training should be aborted and the training state should be set to ready_for_cleanup."""
    trainer = test_initialized_trainer

    create_active_training_file(trainer, training_state=TrainerState.ConfusionMatrixSynced)
    trainer._init_from_last_training()

    _ = asyncio.get_running_loop().create_task(trainer._run())

    await assert_training_state(trainer.training, TrainerState.TrainModelUploading, timeout=1, interval=0.001)
    # TODO goes to finished because of the error
    await assert_training_state(trainer.training, TrainerState.ReadyForCleanup, timeout=2, interval=0.001)

    assert trainer_has_error(trainer)
    assert trainer.training.training_state == TrainerState.ReadyForCleanup
    assert trainer.training.model_uuid_for_detecting is None
    assert trainer.node.last_training_io.load() == trainer.training


async def test_mock_loop_response_example(mocker: MockerFixture, test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer

    mock_upload_model_for_training(mocker, 'new_model_id')

    create_active_training_file(trainer)
    trainer._init_from_last_training()

    # pylint: disable=protected-access
    await trainer._upload_model_return_new_model_uuid(Context(organization='zauberzeug', project='demo'))


def mock_upload_model_for_training(mocker, return_value):
    mocker.patch('learning_loop_node.data_exchanger.DataExchanger.upload_model_get_uuid', return_value=return_value)
