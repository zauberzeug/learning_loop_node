import asyncio

from pytest_mock import MockerFixture  # pip install pytest-mock

from learning_loop_node.trainer.tests.state_helper import (
    assert_training_state, create_active_training_file)
from learning_loop_node.trainer.tests.testing_trainer import TestingTrainer
from learning_loop_node.trainer.trainer_logic import TrainerLogic
from learning_loop_node.trainer.trainer_node import TrainerNode

error_key = 'sync_confusion_matrix'


def trainer_has_error(trainer: TrainerLogic):
    return trainer.errors.has_error_for(error_key)


async def test_nothing_to_sync(test_initialized_trainer: TestingTrainer):
    trainer = test_initialized_trainer

    create_active_training_file(trainer, training_state='training_finished')
    trainer.load_active_training()

    _ = asyncio.get_running_loop().create_task(trainer.train())

    await assert_training_state(trainer.training, 'confusion_matrix_synced', timeout=1, interval=0.001)
    assert trainer_has_error(trainer) is False
    assert trainer.training.training_state == 'confusion_matrix_synced'
    assert trainer.node.last_training_io.load() == trainer.training


async def test_unsynced_model_available__sync_successful(test_initialized_trainer_node: TrainerNode, mocker: MockerFixture):
    trainer = test_initialized_trainer_node.trainer
    assert isinstance(trainer, TestingTrainer)

    await mock_socket_io_call(mocker, test_initialized_trainer_node, {'success': True})
    create_active_training_file(trainer, training_state='training_finished')

    trainer.load_active_training()
    trainer.has_new_model = True

    _ = asyncio.get_running_loop().create_task(trainer.train())
    await assert_training_state(trainer.training, 'confusion_matrix_synced', timeout=1, interval=0.001)

    assert trainer_has_error(trainer) is False
    assert trainer.training.training_state == 'confusion_matrix_synced'
    assert trainer.node.last_training_io.load() == trainer.training


async def test_unsynced_model_available__sio_not_connected(test_initialized_trainer_node: TrainerNode):
    trainer = test_initialized_trainer_node.trainer
    assert isinstance(trainer, TestingTrainer)

    create_active_training_file(trainer, training_state='training_finished')
    trainer.load_active_training()

    assert test_initialized_trainer_node.sio_client.connected is False
    trainer.has_new_model = True

    _ = asyncio.get_running_loop().create_task(trainer.train())

    await assert_training_state(trainer.training, 'confusion_matrix_syncing', timeout=1, interval=0.001)
    await assert_training_state(trainer.training, 'training_finished', timeout=1, interval=0.001)

    assert trainer_has_error(trainer)
    assert trainer.training.training_state == 'training_finished'
    assert trainer.node.last_training_io.load() == trainer.training


async def test_unsynced_model_available__request_is_not_successful(test_initialized_trainer_node: TrainerNode, mocker: MockerFixture):
    trainer = test_initialized_trainer_node.trainer
    assert isinstance(trainer, TestingTrainer)

    await mock_socket_io_call(mocker, test_initialized_trainer_node, {'success': False})

    create_active_training_file(trainer, training_state='training_finished')
    trainer.load_active_training()

    trainer.has_new_model = True
    _ = asyncio.get_running_loop().create_task(trainer.train())

    await assert_training_state(trainer.training, 'confusion_matrix_syncing', timeout=1, interval=0.001)
    await assert_training_state(trainer.training, 'training_finished', timeout=1, interval=0.001)

    assert trainer_has_error(trainer)
    assert trainer.training.training_state == 'training_finished'
    assert trainer.node.last_training_io.load() == trainer.training


async def test_basic_mock(test_initialized_trainer_node: TrainerNode, mocker: MockerFixture):
    node = test_initialized_trainer_node

    patched_call_return_value = asyncio.Future()
    patched_call_return_value.set_result({'success': True})
    mocker.patch.object(node.sio_client, 'call', return_value=patched_call_return_value)
    assert await (await node.sio_client.call()) == {'success': True}  # type: ignore


async def mock_socket_io_call(mocker, trainer_node: TrainerNode, return_value):
    for _ in range(10):
        if trainer_node.sio_client is None:
            await asyncio.sleep(0.1)
        else:
            break
    else:
        raise Exception('sio_client is not available ' + str(trainer_node.sio_client))
    mocker.patch.object(trainer_node.sio_client, 'call', return_value=return_value)
