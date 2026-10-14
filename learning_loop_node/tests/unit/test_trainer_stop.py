import asyncio
import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ...data_classes import Context
from ...enums import TrainerState
from ...globals import GLOBALS
from ...trainer.rest import backdoor_controls
from ...trainer.trainer_node import TrainerNode
from ..trainer.testing_trainer_logic import TestingTrainerLogic

# pylint: disable=protected-access


@pytest.mark.parametrize('state', [TrainerState.TrainingFinished, TrainerState.ConfusionMatrixSyncing,
                                   TrainerState.ConfusionMatrixSynced, TrainerState.TrainModelUploading,
                                   TrainerState.TrainModelUploaded, TrainerState.Detecting, TrainerState.Detected,
                                   TrainerState.DetectionUploading, TrainerState.ReadyForCleanup])
async def test_a_stop_after_the_training_has_run_lets_its_upload_finish(trainer: TestingTrainerLogic,
                                                                        state: TrainerState):
    trainer.training.training_state = state

    await trainer.stop()

    assert not trainer.training_task.done()


async def test_a_stop_after_the_training_process_ended_lets_its_upload_finish(trainer: TestingTrainerLogic):
    trainer.training.training_state = TrainerState.TrainingRunning

    stop = asyncio.create_task(trainer.stop())
    await asyncio.sleep(0.3)
    assert not stop.done()

    trainer.training.training_state = TrainerState.TrainingFinished
    await asyncio.wait_for(stop, timeout=1)

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


async def test_a_reset_aborts_even_the_upload(trainer: TestingTrainerLogic, monkeypatch: pytest.MonkeyPatch):
    trainer.training.training_state = TrainerState.TrainModelUploading
    monkeypatch.setattr(trainer.node, 'reconnect_to_loop', AsyncMock())

    await backdoor_controls.reset(SimpleNamespace(app=trainer.node))

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
    with contextlib.suppress(asyncio.CancelledError):
        await trainer.training_task
