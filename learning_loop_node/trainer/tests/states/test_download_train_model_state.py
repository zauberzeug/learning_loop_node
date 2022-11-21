
from learning_loop_node.trainer.tests.testing_trainer import TestingTrainer
from learning_loop_node.trainer.trainer import Trainer
import asyncio
from learning_loop_node.trainer.tests.states.state_helper import assert_training_state
from learning_loop_node.trainer.tests.states import state_helper
from learning_loop_node.trainer import active_training
import os
import logging


async def test_downloading_is_successfull():
    state_helper.create_active_training_file(training_state='data_downloaded')
    trainer = Trainer(model_format='mocked')
    trainer.training = active_training.load()  # normally done by node

    download_task = asyncio.get_running_loop().create_task(trainer.download_model())
    await assert_training_state(trainer.training, 'train_model_downloading', timeout=1, interval=0.001)
    await assert_training_state(trainer.training, 'train_model_downloaded', timeout=1, interval=0.001)

    assert trainer.training.training_state == 'train_model_downloaded'
    assert active_training.load() == trainer.training

    # file on disk
    assert os.path.exists(f'{trainer.training.training_folder}/base_model.json')
    assert os.path.exists(f'{trainer.training.training_folder}/file_1.txt')
    assert os.path.exists(f'{trainer.training.training_folder}/file_2.txt')


async def test_abort_download_model():
    state_helper.create_active_training_file(training_state='data_downloaded')
    trainer = TestingTrainer()
    trainer.training = active_training.load()  # normally done by node

    train_task = asyncio.get_running_loop().create_task(trainer.train(None, None))
    await assert_training_state(trainer.training, 'train_model_downloading', timeout=1, interval=0.001)

    trainer.stop()
    await asyncio.sleep(0.1)

    assert trainer.training == None
    assert active_training.exists() == False


async def test_downloading_failed():
    state_helper.create_active_training_file(training_state='data_downloaded',
                                             base_model_id='00000000-0000-0000-0000-000000000000')  # bad model id)
    trainer = Trainer(model_format='mocked')
    trainer.training = active_training.load()  # normally done by node

    train_task = asyncio.get_running_loop().create_task(trainer.train(None, None))
    await assert_training_state(trainer.training, 'train_model_downloading', timeout=1, interval=0.001)
    await assert_training_state(trainer.training, 'data_downloaded', timeout=1, interval=0.001)

    assert trainer.errors.has_error_for('download_model')
    assert trainer.training is not None
    assert trainer.training.training_state == 'data_downloaded'
    assert active_training.load() == trainer.training
