
import asyncio
import os

from learning_loop_node.trainer.tests.state_helper import (
    assert_training_state, create_active_training_file)
from learning_loop_node.trainer.tests.testing_trainer_logic import \
    TestingTrainerLogic


async def test_downloading_is_successful(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer
    create_active_training_file(trainer, training_state='data_downloaded')

    trainer.model_format = 'mocked'
    trainer.load_last_training()

    _ = asyncio.get_running_loop().create_task(trainer.download_model())
    await assert_training_state(trainer.training, 'train_model_downloading', timeout=1, interval=0.001)
    await assert_training_state(trainer.training, 'train_model_downloaded', timeout=1, interval=0.001)

    assert trainer.training.training_state == 'train_model_downloaded'
    assert trainer.node.last_training_io.load() == trainer.training

    # file on disk
    assert os.path.exists(f'{trainer.training.training_folder}/base_model.json')
    assert os.path.exists(f'{trainer.training.training_folder}/file_1.txt')
    assert os.path.exists(f'{trainer.training.training_folder}/file_2.txt')


async def test_abort_download_model(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer
    create_active_training_file(trainer, training_state='data_downloaded')
    trainer.load_last_training()

    _ = asyncio.get_running_loop().create_task(trainer.run())
    await assert_training_state(trainer.training, 'train_model_downloading', timeout=1, interval=0.001)

    await trainer.stop()
    await asyncio.sleep(0.1)

    assert trainer._training is None
    assert trainer.node.last_training_io.exists() is False


async def test_downloading_failed(test_initialized_trainer: TestingTrainerLogic):
    trainer = test_initialized_trainer
    create_active_training_file(trainer, training_state='data_downloaded',
                                base_model_id='00000000-0000-0000-0000-000000000000')  # bad model id)
    trainer.load_last_training()

    _ = asyncio.get_running_loop().create_task(trainer.run())
    await assert_training_state(trainer.training, 'train_model_downloading', timeout=1, interval=0.001)
    await assert_training_state(trainer.training, 'data_downloaded', timeout=1, interval=0.001)

    assert trainer.errors.has_error_for('download_model')
    assert trainer._training is not None  # pylint: disable=protected-access
    assert trainer.training.training_state == 'data_downloaded'
    assert trainer.node.last_training_io.load() == trainer.training
