from learning_loop_node.trainer.tests.testing_trainer import TestingTrainer
import asyncio
from learning_loop_node.trainer.tests.states.state_helper import assert_training_state
from learning_loop_node.trainer.tests.states import state_helper
from learning_loop_node.trainer import active_training
from learning_loop_node.rest.downloads import DownloadError
import pytest


async def test_successfull_detecting(mocker):
    state_helper.create_active_training_file(training_state='train_model_uploaded',
                                             model_id_for_detecting='7f5eabb4-227a-e7c7-8f0b-f825cc47340d')  # version 1.2 of demo project
    trainer = TestingTrainer()
    trainer.training = active_training.load()  # normally done by node

    detect_task = asyncio.get_running_loop().create_task(trainer.do_detections())

    await assert_training_state(trainer.training, 'detecting', timeout=1, interval=0.001)
    await assert_training_state(trainer.training, 'detected', timeout=1, interval=0.001)

    assert trainer.training.training_state == 'detected'
    assert active_training.load() == trainer.training
    assert active_training.detections_exist(trainer.training)


async def test_detecting_can_be_aborted():
    state_helper.create_active_training_file(training_state='train_model_uploaded',
                                             model_id_for_detecting='7f5eabb4-227a-e7c7-8f0b-f825cc47340d')  # version 1.2 of demo project
    trainer = TestingTrainer()
    trainer.training = active_training.load()  # normally done by node
    training = trainer.training

    train_task = asyncio.get_running_loop().create_task(trainer.train(None, None))

    await assert_training_state(trainer.training, 'detecting', timeout=1, interval=0.001)
    trainer.stop()
    await asyncio.sleep(0.1)

    assert trainer.training is None
    assert active_training.detections_exist(training) is False
    assert active_training.exists() == False


async def test_model_not_downloadable_error(mocker):
    state_helper.create_active_training_file(training_state='train_model_uploaded',
                                             model_id_for_detecting='00000000-0000-0000-0000-000000000000')  # bad model id
    trainer = TestingTrainer()
    trainer.training = active_training.load()  # normally done by node

    train_task = asyncio.get_running_loop().create_task(trainer.train(None, None))

    await assert_training_state(trainer.training, 'detecting', timeout=1, interval=0.001)
    await assert_training_state(trainer.training, 'train_model_uploaded', timeout=1, interval=0.001)

    assert trainer.training.training_state == 'train_model_uploaded'
    assert trainer.training.model_id_for_detecting == '00000000-0000-0000-0000-000000000000'
    assert active_training.load() == trainer.training


def test_save_load_detections():
    detections = [{'some_key': 'some_value'}]
    state_helper.create_active_training_file()
    trainer = TestingTrainer()
    trainer.training = active_training.load()

    active_training.save_detections(trainer.training, detections)
    assert active_training.detections_exist(trainer.training)

    stored_detections = active_training.load_detections(trainer.training)
    assert stored_detections == detections

    active_training.delete_detections(trainer.training)
    assert active_training.detections_exist(trainer.training) is False