import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from ...data_classes import Context
from ...enums import TrainerState
from ...globals import GLOBALS
from ...trainer.trainer_node import TrainerNode
from ..trainer.testing_trainer_logic import TestingTrainerLogic

# pylint: disable=protected-access


@pytest.mark.parametrize('state', [TrainerState.TrainingFinished, TrainerState.ConfusionMatrixSynced,
                                   TrainerState.TrainModelUploading])
async def test_a_stop_before_the_model_is_uploaded_lets_the_upload_finish(trainer: TestingTrainerLogic,
                                                                          state: TrainerState):
    trainer.training.training_state = state

    await trainer.stop()

    assert not trainer.training_task.done()


@pytest.mark.parametrize('state', [TrainerState.DataDownloading, TrainerState.TrainModelUploaded,
                                   TrainerState.Detecting, TrainerState.DetectionUploading])
async def test_a_stop_before_the_training_or_after_the_upload_aborts(trainer: TestingTrainerLogic,
                                                                     state: TrainerState):
    trainer.training.training_state = state

    await trainer.stop()

    assert trainer.training_task.cancelled()


async def test_a_stop_on_shutdown_aborts_even_the_upload(trainer: TestingTrainerLogic):
    trainer.training.training_state = TrainerState.TrainModelUploading
    trainer.shutdown_event.set()

    await trainer.stop()

    assert trainer.training_task.cancelled()


@pytest.fixture(name='trainer')
async def trainer_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[TestingTrainerLogic]:
    """A trainer whose training task only waits, so a test sees whether a stop cancels it."""
    monkeypatch.setattr(GLOBALS, 'data_folder', str(tmp_path))
    monkeypatch.delenv('RESTART_AFTER_TRAINING', raising=False)
    trainer = TestingTrainerLogic()
    TrainerNode(name='test', trainer_logic=trainer, uuid='NODE-000-0000-0000-0000-000000000000')
    trainer._init_new_training(Context(organization='zauberzeug', project='demo'),
                               {'categories': [], 'training_number': 0, 'model_variant': '', 'hyperparameters': {}})
    trainer.training_task = asyncio.create_task(asyncio.sleep(10))
    yield trainer
    trainer.training_task.cancel()
