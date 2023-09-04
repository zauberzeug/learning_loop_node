
import asyncio
import os

from learning_loop_node.data_classes import Context
from learning_loop_node.trainer.tests.states.state_helper import (
    assert_training_state, create_active_training_file)
from learning_loop_node.trainer.tests.testing_trainer import TestingTrainer
from learning_loop_node.trainer.trainer import Trainer


async def test_downloading_is_successful():
    create_active_training_file(training_state='data_downloaded')

    trainer = Trainer(model_format='mocked')
    trainer.init(context=Context(organization='zauberzeug', project='demo'),
                 details={}, node_uuid='00000000-0000-0000-0000-000000000000')
    trainer.load_active_training()

    assert trainer._training is not None and trainer._active_training_io is not None

    _ = asyncio.get_running_loop().create_task(trainer.download_model())
    await assert_training_state(trainer._training, 'train_model_downloading', timeout=1, interval=0.001)
    await assert_training_state(trainer._training, 'train_model_downloaded', timeout=1, interval=0.001)

    assert trainer._training.training_state == 'train_model_downloaded'
    assert trainer._active_training_io.load() == trainer._training

    # file on disk
    assert os.path.exists(f'{trainer._training.training_folder}/base_model.json')
    assert os.path.exists(f'{trainer._training.training_folder}/file_1.txt')
    assert os.path.exists(f'{trainer._training.training_folder}/file_2.txt')


async def test_abort_download_model(test_initialized_trainer: TestingTrainer):
    trainer = test_initialized_trainer
    assert trainer._training is not None and trainer._active_training_io is not None

    create_active_training_file(training_state='data_downloaded')
    trainer.load_active_training()

    _ = asyncio.get_running_loop().create_task(trainer.train(None, None))
    await assert_training_state(trainer._training, 'train_model_downloading', timeout=1, interval=0.001)

    trainer.stop()
    await asyncio.sleep(0.1)

    assert trainer._training is None
    assert trainer._active_training_io.exists() is False


async def test_downloading_failed(test_initialized_trainer: TestingTrainer):
    trainer = test_initialized_trainer
    assert trainer._training is not None and trainer._active_training_io is not None

    create_active_training_file(training_state='data_downloaded',
                                base_model_id='00000000-0000-0000-0000-000000000000')  # bad model id)
    trainer.load_active_training()

    assert trainer._training is not None and trainer._active_training_io is not None

    _ = asyncio.get_running_loop().create_task(trainer.train(None, None))
    await assert_training_state(trainer._training, 'train_model_downloading', timeout=1, interval=0.001)
    await assert_training_state(trainer._training, 'data_downloaded', timeout=1, interval=0.001)

    assert trainer.errors.has_error_for('download_model')
    assert trainer._training is not None
    assert trainer._training.training_state == 'data_downloaded'
    assert trainer._active_training_io.load() == trainer._training
