
from learning_loop_node.trainer.trainer import Trainer
import asyncio
from learning_loop_node.trainer.tests.states.state_helper import assert_training_file
from learning_loop_node.trainer.tests.states import state_helper
from learning_loop_node.trainer import active_training
from learning_loop_node.trainer.training import Training
import os
import logging


async def test_successful_downloading():
    def _assert_training_contains_all_data(training: Training) -> None:
        assert training.training_state == 'train_model_downloaded'
        # file on disk
        assert os.path.exists(f'{training.training_folder}/base_model.json')
        assert os.path.exists(f'{training.training_folder}/file_1.txt')
        assert os.path.exists(f'{training.training_folder}/file_2.txt')

    state_helper.create_active_training_file()
    trainer = Trainer(model_format='mocked')
    trainer.training = active_training.load()  # normally done by node

    await trainer.prepare()
    await trainer.download_model()

    _assert_training_contains_all_data(trainer.training)
    assert_training_file(exists=True)

    loaded_training = active_training.load()
    _assert_training_contains_all_data(loaded_training)
