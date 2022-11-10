from learning_loop_node.trainer.tests.testing_trainer import TestingTrainer
from learning_loop_node.trainer.trainer import Trainer
import asyncio
from learning_loop_node.trainer.tests.states.state_helper import assert_training_file, assert_training_state
from learning_loop_node.trainer.tests.states import state_helper
from learning_loop_node.trainer import active_training
from learning_loop_node.trainer.training import Training
import os
import logging
from learning_loop_node.status import State
from learning_loop_node.trainer.trainer_node import TrainerNode


async def test_nothing_to_sync():
    state_helper.create_active_training_file()
    trainer = TestingTrainer()
    trainer.training = active_training.load()  # normally done by node

    await trainer.prepare()
    await trainer.download_model()
    train_task = asyncio.get_running_loop().create_task(trainer.run_training())

    await assert_training_state(trainer.training, 'training_running', timeout=1, interval=0.001)
    trainer.executor.stop()  # NOTE normally a training terminates itself e.g

    await trainer.ensure_confusion_matrix_synced(trainer_node_uuid=None, sio_client=None)
    assert trainer.training.training_state == 'confusion_matrix_synced'


async def test_unsynced_model_available__sync_successfull(test_trainer_node: TrainerNode, mocker):
    patched_call_return_value = asyncio.Future()
    patched_call_return_value.set_result(
        {'success': True})
    mocker.patch.object(test_trainer_node.sio_client, 'call', return_value=patched_call_return_value)

    state_helper.create_active_training_file()

    await test_trainer_node.connect()
    trainer = test_trainer_node.trainer
    trainer.training = active_training.load()  # normally done by node

    await trainer.prepare()
    await trainer.download_model()
    train_task = asyncio.get_running_loop().create_task(trainer.run_training())

    await assert_training_state(trainer.training, 'training_running', timeout=1, interval=0.001)
    trainer.executor.stop()  # NOTE normally a training terminates itself e.g

    trainer.has_new_model = True
    await trainer.ensure_confusion_matrix_synced(test_trainer_node.uuid, sio_client=test_trainer_node.sio_client)
    assert trainer.training.training_state == 'confusion_matrix_synced'
    await train_task


async def test_unsynced_model_available_but_sio_connection_not_connected(test_trainer_node: TrainerNode):
    state_helper.create_active_training_file()

    trainer = test_trainer_node.trainer
    trainer.training = active_training.load()  # normally done by node

    await trainer.prepare()
    await trainer.download_model()
    train_task = asyncio.get_running_loop().create_task(trainer.run_training())

    await assert_training_state(trainer.training, 'training_running', timeout=1, interval=0.001)
    trainer.executor.stop()  # NOTE normally a training terminates itself e.g
    await asyncio.sleep(0.1)
    assert trainer.training.training_state == 'training_finished'

    trainer.has_new_model = True
    await trainer.ensure_confusion_matrix_synced(test_trainer_node.uuid, sio_client=test_trainer_node.sio_client)

    # syncronization failed, State is still 'training_finished'
    assert trainer.training.training_state == 'training_finished'
    await train_task


async def test_unsynced_model_available_but_request_is_not_successful(test_trainer_node: TrainerNode, mocker):
    patched_call_return_value = asyncio.Future()
    patched_call_return_value.set_result(
        {'success': False})
    mocker.patch.object(test_trainer_node.sio_client, 'call', return_value=patched_call_return_value)

    state_helper.create_active_training_file()

    trainer = test_trainer_node.trainer
    trainer.training = active_training.load()  # normally done by node

    await trainer.prepare()
    await trainer.download_model()
    train_task = asyncio.get_running_loop().create_task(trainer.run_training())

    await assert_training_state(trainer.training, 'training_running', timeout=1, interval=0.001)
    trainer.executor.stop()  # NOTE normally a training terminates itself e.g
    await asyncio.sleep(0.1)
    assert trainer.training.training_state == 'training_finished'

    trainer.has_new_model = True
    await trainer.ensure_confusion_matrix_synced(test_trainer_node.uuid, sio_client=test_trainer_node.sio_client)

    # syncronization failed, State is still 'training_finished'
    assert trainer.training.training_state == 'training_finished'
    await train_task


async def test_basic_mock(test_trainer_node: TrainerNode, mocker):
    patched_call_return_value = asyncio.Future()
    patched_call_return_value.set_result(
        {'success': True})
    mocker.patch.object(test_trainer_node.sio_client, 'call', return_value=patched_call_return_value)
    assert await test_trainer_node.sio_client.call() == {'success': True}
