import asyncio

from pytest_mock import MockerFixture

from learning_loop_node.data_classes import Context
from learning_loop_node.trainer.tests.state_helper import (
    assert_training_state, create_active_training_file)
from learning_loop_node.trainer.tests.testing_trainer_logic import \
    TestingTrainerLogic
from learning_loop_node.trainer.trainer_logic import TrainerLogic

error_key = 'upload_model'


def trainer_has_error(trainer: TrainerLogic):
    return trainer.errors.has_error_for(error_key)


async def test_successful_upload(mocker: MockerFixture, test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer
    mock_upload_model_for_training(mocker, {'id': 'some_id'})

    create_active_training_file(trainer)
    trainer.load_active_training()

    train_task = asyncio.get_running_loop().create_task(trainer.upload_model())

    await assert_training_state(trainer.training, 'train_model_uploading', timeout=1, interval=0.001)
    await train_task

    assert trainer_has_error(trainer) is False
    assert trainer.training.training_state == 'train_model_uploaded'
    assert trainer.training.model_id_for_detecting == 'some_id'
    assert trainer.node.last_training_io.load() == trainer.training


async def test_abort_upload_model(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer

    create_active_training_file(trainer, training_state='confusion_matrix_synced')
    trainer.load_active_training()

    _ = asyncio.get_running_loop().create_task(trainer.train())

    await assert_training_state(trainer.training, 'train_model_uploading', timeout=1, interval=0.001)

    await trainer.stop()
    await asyncio.sleep(0.1)

    assert trainer._training is None  # pylint: disable=protected-access
    assert trainer.node.last_training_io.exists() is False


async def test_bad_server_response_content(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer

    # without a running training the loop will not allow uploading.
    create_active_training_file(trainer, training_state='confusion_matrix_synced')
    trainer.load_active_training()

    _ = asyncio.get_running_loop().create_task(trainer.train())

    await assert_training_state(trainer.training, 'train_model_uploading', timeout=1, interval=0.001)
    await assert_training_state(trainer.training, 'confusion_matrix_synced', timeout=2, interval=0.001)

    assert trainer_has_error(trainer)
    assert trainer.training.training_state == 'confusion_matrix_synced'
    assert trainer.training.model_id_for_detecting is None
    assert trainer.node.last_training_io.load() == trainer.training


async def test_mock_loop_response_example(mocker: MockerFixture, test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer

    mock_upload_model_for_training(mocker, {'some_key': 'some_data'})

    create_active_training_file(trainer)
    trainer.load_active_training()

    # pylint: disable=protected-access
    result = await trainer._upload_model(Context(organization='zauberzeug', project='demo'))

    assert result['some_key'] == 'some_data'


def mock_upload_model_for_training(mocker, return_value):
    mocker.patch('learning_loop_node.data_exchanger.DataExchanger.upload_model_for_training', return_value=return_value)
