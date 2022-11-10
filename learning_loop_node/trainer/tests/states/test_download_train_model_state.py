
from learning_loop_node.trainer.trainer import Trainer
import asyncio
from learning_loop_node.trainer.tests.states.state_helper import assert_training_file, assert_training_state
from learning_loop_node.trainer.tests.states import state_helper
from learning_loop_node.trainer import active_training
from learning_loop_node.trainer.training import Training
import os
import logging


async def test_downloading_is_successfull():
    state_helper.create_active_training_file(training_state='some_previous_state')
    trainer = Trainer(model_format='mocked')
    trainer.training = active_training.load()  # normally done by node

    await trainer.download_model()

    assert trainer.training.training_state == 'train_model_downloaded'
    assert active_training.load() == trainer.training

    # file on disk
    assert os.path.exists(f'{trainer.training.training_folder}/base_model.json')
    assert os.path.exists(f'{trainer.training.training_folder}/file_1.txt')
    assert os.path.exists(f'{trainer.training.training_folder}/file_2.txt')


async def test_abort_download_model():
    state_helper.create_active_training_file(training_state='some_previous_state')
    trainer = Trainer(model_format='mocked')
    trainer.training = active_training.load()  # normally done by node

    download_task = asyncio.get_running_loop().create_task(trainer.download_model())

    await assert_training_state(trainer.training, 'train_model_downloading', timeout=1, interval=0.001)
    trainer.stop()
    await asyncio.sleep(0.1)

    assert trainer.training == None
    assert active_training.load().training_state == 'ready_for_cleanup'


async def test_downloading_failed():
    state_helper.create_active_training_file(training_state='some_previous_state',
                                             base_model_id='00000000-0000-0000-0000-000000000000')  # bad model id)
    trainer = Trainer(model_format='mocked')
    trainer.training = active_training.load()  # normally done by node

    download_task = asyncio.get_running_loop().create_task(trainer.download_model())
    await assert_training_state(trainer.training, 'train_model_downloading', timeout=1, interval=0.001)
    await download_task

    assert trainer.training is not None
    assert trainer.training.training_state == 'some_previous_state'
    assert active_training.load() == trainer.training
