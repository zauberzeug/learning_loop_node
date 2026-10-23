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
from ..test_helper import condition
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


async def test_a_stop_during_the_training_ends_its_process(trainer: TestingTrainerLogic):
    train = asyncio.create_task(trainer._train())
    await condition(lambda: trainer._executor is not None and trainer._executor.is_running(), timeout=2, interval=0.01)

    await trainer.stop()
    await asyncio.wait_for(train, timeout=2)

    assert not trainer.executor.is_running()
    assert not trainer.training_task.done()


async def test_a_stop_after_the_training_process_ended_lets_its_upload_finish(trainer: TestingTrainerLogic):
    trainer.training.training_state = TrainerState.TrainingRunning
    trainer._process_started = True  # the process has ended, but `_train` has not handed over yet

    await asyncio.wait_for(trainer.stop(), timeout=1)

    assert not trainer.training_task.done()


async def test_a_stopped_training_winds_down_when_its_process_fails(trainer: TestingTrainerLogic):
    train = asyncio.create_task(trainer._perform_state('run_training', TrainerState.TrainingRunning,
                                                       TrainerState.TrainingFinished, trainer._train))
    await condition(lambda: trainer._executor is not None and trainer._executor.is_running(), timeout=2, interval=0.01)
    trainer.error_msg = 'graphics card is out of memory'

    await trainer.stop()
    await asyncio.wait_for(train, timeout=2)

    assert trainer.training.training_state == TrainerState.TrainingFinished


async def test_a_stop_while_the_upload_fails_aborts_it(trainer: TestingTrainerLogic):
    trainer.training.training_state = TrainerState.ConfusionMatrixSynced
    await trainer._perform_state('upload_model', TrainerState.TrainModelUploading, TrainerState.TrainModelUploaded,
                                 AsyncMock(side_effect=ConnectionError('loop is unreachable')))

    await trainer.stop()

    assert trainer.training_task.cancelled()


async def test_a_stop_after_the_upload_recovered_lets_it_finish(trainer: TestingTrainerLogic):
    trainer.training.training_state = TrainerState.ConfusionMatrixSynced
    await trainer._perform_state('upload_model', TrainerState.TrainModelUploading, TrainerState.TrainModelUploaded,
                                 AsyncMock(side_effect=ConnectionError('loop is unreachable')))
    await trainer._perform_state('upload_model', TrainerState.TrainModelUploading, TrainerState.TrainModelUploaded,
                                 AsyncMock())

    await trainer.stop()

    assert not trainer.training_task.done()


@pytest.mark.parametrize('state', [TrainerState.DataDownloading, TrainerState.TrainModelDownloaded,
                                   TrainerState.TrainingRunning])
async def test_a_stop_before_the_training_has_run_aborts_it(trainer: TestingTrainerLogic, state: TrainerState):
    trainer.training.training_state = state

    await trainer.stop()

    assert trainer.training_task.cancelled()


async def test_a_stop_on_shutdown_aborts_even_the_upload(trainer: TestingTrainerLogic):
    trainer.training.training_state = TrainerState.TrainModelUploading
    trainer.shutdown_event.set()

    await trainer.stop()

    assert trainer.training_task.cancelled()


async def test_a_shutdown_aborts_even_the_upload(trainer: TestingTrainerLogic):
    trainer.training.training_state = TrainerState.TrainModelUploading

    await trainer.on_shutdown()

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
    if trainer._executor:
        await trainer._executor.stop_and_wait()
    trainer.training_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await trainer.training_task
