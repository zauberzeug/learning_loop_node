import asyncio

import pytest

from ...data_classes import Context
from ...enums import TrainerState
from ...globals import GLOBALS
from ...trainer.trainer_node import TrainerNode
from ..trainer.testing_trainer_logic import TestingTrainerLogic

# pylint: disable=protected-access


@pytest.fixture
async def trainer(tmp_path, monkeypatch):
    monkeypatch.setattr(GLOBALS, 'data_folder', str(tmp_path))
    monkeypatch.delenv('RESTART_AFTER_TRAINING', raising=False)
    trainer = TestingTrainerLogic()
    TrainerNode(name='test', trainer_logic=trainer, uuid='NODE-000-0000-0000-0000-000000000000')
    trainer._init_new_training(Context(organization='zauberzeug', project='demo'),
                               {'categories': [], 'training_number': 0, 'model_variant': '', 'hyperparameters': {}})
    trainer.training_task = asyncio.create_task(asyncio.sleep(10))
    yield trainer
    trainer.training_task.cancel()


@pytest.mark.parametrize('state', [TrainerState.TrainingFinished, TrainerState.TrainModelUploading,
                                   TrainerState.Detecting])
async def test_a_stop_after_the_training_has_run_lets_its_upload_finish(trainer: TestingTrainerLogic, state):
    trainer.training.training_state = state

    await trainer.stop()

    assert not trainer.training_task.done()


async def test_a_stop_before_the_training_has_run_aborts_it(trainer: TestingTrainerLogic):
    trainer.training.training_state = TrainerState.DataDownloading

    await trainer.stop()

    assert trainer.training_task.cancelled()


async def test_a_stop_on_shutdown_aborts_even_the_upload(trainer: TestingTrainerLogic):
    trainer.training.training_state = TrainerState.TrainModelUploading
    trainer.shutdown_event.set()

    await trainer.stop()

    assert trainer.training_task.cancelled()
