from learning_loop_node.trainer.tests.testing_trainer import TestingTrainer
import asyncio
from learning_loop_node.trainer.tests.states.state_helper import assert_training_state
from learning_loop_node.trainer.tests.states import state_helper
from learning_loop_node.trainer import active_training
from learning_loop_node.trainer.trainer import Trainer
from learning_loop_node.context import Context


error_key = 'upload_model'


def trainer_has_error(trainer: Trainer):
    return trainer.errors.has_error_for(error_key)


async def test_successful_upload(mocker):
    mock_upload_model_for_training(mocker, {'id': 'some_id'})

    state_helper.create_active_training_file()
    trainer = TestingTrainer()
    trainer.training = active_training.load()  # normally done by node

    train_task = asyncio.get_running_loop().create_task(trainer.upload_model())

    await assert_training_state(trainer.training, 'train_model_uploading', timeout=1, interval=0.001)
    await train_task

    assert trainer_has_error(trainer) == False
    assert trainer.training.training_state == 'train_model_uploaded'
    assert trainer.training.model_id_for_detecting == 'some_id'
    assert active_training.load() == trainer.training


async def test_abort_upload_model():
    state_helper.create_active_training_file(training_state='confusion_matrix_synced')
    trainer = TestingTrainer()
    trainer.training = active_training.load()  # normally done by node

    train_task = asyncio.get_running_loop().create_task(trainer.train(None, None))

    await assert_training_state(trainer.training, 'train_model_uploading', timeout=1, interval=0.001)

    trainer.stop()
    await asyncio.sleep(0.1)

    assert trainer.training == None
    assert active_training.exists() == False


async def test_bad_server_response_content(mocker):
    # without a running training the loop will not allow uploading.
    state_helper.create_active_training_file(training_state='confusion_matrix_synced')
    trainer = TestingTrainer()
    trainer.training = active_training.load()  # normally done by node

    train_task = asyncio.get_running_loop().create_task(trainer.train(None, None))

    await assert_training_state(trainer.training, 'train_model_uploading', timeout=1, interval=0.001)
    await assert_training_state(trainer.training, 'confusion_matrix_synced', timeout=2, interval=0.001)

    assert trainer_has_error(trainer)
    assert trainer.training.training_state == 'confusion_matrix_synced'
    assert trainer.training.model_id_for_detecting == None
    assert active_training.load() == trainer.training


async def test_mock_loop_response_example(mocker):
    mock_upload_model_for_training(mocker, 'some_data')

    state_helper.create_active_training_file()
    trainer = TestingTrainer()
    trainer.training = active_training.load()  # normally done by node

    result = await trainer._upload_model(Context(organization='zauberzeug', project='demo'))

    assert result == 'some_data'


def mock_upload_model_for_training(mocker, return_value):
    patched_call_return_value = asyncio.Future()
    patched_call_return_value.set_result(
        return_value)
    mocker.patch('learning_loop_node.rest.uploads.upload_model_for_training', return_value=patched_call_return_value)
