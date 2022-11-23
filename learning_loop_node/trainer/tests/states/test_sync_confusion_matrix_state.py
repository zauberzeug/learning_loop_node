from learning_loop_node.trainer.tests.testing_trainer import TestingTrainer
import asyncio
from learning_loop_node.trainer.tests.states.state_helper import assert_training_state
from learning_loop_node.trainer.tests.states import state_helper
from learning_loop_node.trainer import active_training
from learning_loop_node.trainer.trainer import Trainer
from learning_loop_node.trainer.trainer_node import TrainerNode

error_key = 'sync_confusion_matrix'


def trainer_has_error(trainer: Trainer):
    return trainer.errors.has_error_for(error_key)


async def test_nothing_to_sync():
    state_helper.create_active_training_file(training_state='training_finished')
    trainer = TestingTrainer()
    trainer.training = active_training.load()  # normally done by node

    train_task = asyncio.get_running_loop().create_task(trainer.train(None, None))

    await assert_training_state(trainer.training, 'confusion_matrix_synced', timeout=1, interval=0.001)
    assert trainer_has_error(trainer) is False
    assert trainer.training.training_state == 'confusion_matrix_synced'
    assert active_training.load() == trainer.training


async def test_unsynced_model_available__sync_successful(test_trainer_node: TrainerNode, mocker):
    mock_socket_io_call(mocker, test_trainer_node, {'success': True})

    state_helper.create_active_training_file(training_state='training_finished')
    trainer = test_trainer_node.trainer
    trainer.training = active_training.load()  # normally done by node
    trainer.has_new_model = True

    train_task = asyncio.get_running_loop().create_task(trainer.train(
        uuid=test_trainer_node.uuid, sio_client=test_trainer_node.sio_client))
    await assert_training_state(trainer.training, 'confusion_matrix_synced', timeout=1, interval=0.001)

    assert trainer_has_error(trainer) is False
    assert trainer.training.training_state == 'confusion_matrix_synced'
    assert active_training.load() == trainer.training


async def test_unsynced_model_available__sio_not_connected(test_trainer_node: TrainerNode):
    state_helper.create_active_training_file(training_state='training_finished')
    trainer = test_trainer_node.trainer
    trainer.training = active_training.load()  # normally done by node

    assert test_trainer_node.sio_client.connected is False
    trainer.has_new_model = True
    train_task = asyncio.get_running_loop().create_task(trainer.train(
        uuid=test_trainer_node.uuid, sio_client=test_trainer_node.sio_client))

    await assert_training_state(trainer.training, 'confusion_matrix_syncing', timeout=1, interval=0.001)
    await assert_training_state(trainer.training, 'training_finished', timeout=1, interval=0.001)

    assert trainer_has_error(trainer)
    assert trainer.training.training_state == 'training_finished'
    assert active_training.load() == trainer.training


async def test_unsynced_model_available__request_is_not_successful(test_trainer_node: TrainerNode, mocker):
    mock_socket_io_call(mocker, test_trainer_node, {'success': False})

    state_helper.create_active_training_file(training_state='training_finished')
    trainer = test_trainer_node.trainer
    trainer.training = active_training.load()  # normally done by node

    trainer.has_new_model = True
    train_task = asyncio.get_running_loop().create_task(trainer.train(
        uuid=test_trainer_node.uuid, sio_client=test_trainer_node.sio_client))

    await assert_training_state(trainer.training, 'confusion_matrix_syncing', timeout=1, interval=0.001)
    await assert_training_state(trainer.training, 'training_finished', timeout=1, interval=0.001)

    assert trainer_has_error(trainer)
    assert trainer.training.training_state == 'training_finished'
    assert active_training.load() == trainer.training


async def test_basic_mock(test_trainer_node: TrainerNode, mocker):
    patched_call_return_value = asyncio.Future()
    patched_call_return_value.set_result(
        {'success': True})
    mocker.patch.object(test_trainer_node.sio_client, 'call', return_value=patched_call_return_value)
    assert await test_trainer_node.sio_client.call() == {'success': True}


def mock_socket_io_call(mocker, trainer_node: TrainerNode, return_value):
    patched_call_return_value = asyncio.Future()
    patched_call_return_value.set_result(
        return_value)
    mocker.patch.object(trainer_node.sio_client, 'call', return_value=patched_call_return_value)
