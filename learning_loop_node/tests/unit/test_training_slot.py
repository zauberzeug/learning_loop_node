import asyncio
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ...data_classes import PretrainedModel, TrainingStateData
from ...trainer.trainer_logic_generic import TrainerLogicGeneric
from ...trainer.trainer_node import TrainerNode
from ...trainer.training_slot import (
    ENV_VAR,
    AlwaysFreeSlot,
    FileTrainingSlot,
    training_slot_from_env,
)


@pytest.fixture
def lock_path(tmp_path: Path) -> Path:
    return tmp_path / 'training.lock'


def test_a_free_slot_has_no_holder(lock_path: Path):
    assert FileTrainingSlot(lock_path).holder() is None


def test_the_slot_is_exclusive_and_names_its_holder(lock_path: Path):
    first, second = FileTrainingSlot(lock_path), FileTrainingSlot(lock_path)
    assert first.try_acquire('node a')
    assert first.held
    assert not second.try_acquire('node b')
    assert not second.held
    assert second.holder() == 'node a'


def test_inspecting_a_slot_does_not_create_its_file(lock_path: Path):
    assert FileTrainingSlot(lock_path).holder() is None
    assert not lock_path.exists()


def test_the_file_is_writable_for_sibling_users(lock_path: Path):
    slot = FileTrainingSlot(lock_path)
    old_umask = os.umask(0o022)
    try:
        slot.try_acquire('me')
    finally:
        os.umask(old_umask)
        slot.release()
    assert lock_path.stat().st_mode & 0o777 == 0o666


@pytest.mark.skipif(os.geteuid() == 0, reason='root reads any file')
def test_an_unreadable_slot_counts_as_free(lock_path: Path):
    lock_path.touch(mode=0o000)
    assert FileTrainingSlot(lock_path).holder() is None


def test_release_hands_the_slot_on(lock_path: Path):
    first, second = FileTrainingSlot(lock_path), FileTrainingSlot(lock_path)
    first.try_acquire('node a')
    first.release()
    assert not first.held
    assert second.try_acquire('node b')
    assert first.holder() == 'node b'


def test_acquiring_twice_is_harmless(lock_path: Path):
    slot = FileTrainingSlot(lock_path)
    assert slot.try_acquire('node a')
    assert slot.try_acquire('node a')
    slot.release()
    slot.release()
    assert slot.holder() is None


def test_a_dying_process_frees_the_slot(lock_path: Path):
    held_marker = lock_path.with_suffix('.held')
    other = subprocess.Popen([sys.executable, '-c', f'''
from learning_loop_node.trainer.training_slot import FileTrainingSlot
import pathlib, time
assert FileTrainingSlot({str(lock_path)!r}).try_acquire('other process')
pathlib.Path({str(held_marker)!r}).touch()
time.sleep(60)
'''], stdout=subprocess.DEVNULL)
    try:
        deadline = time.time() + 30
        while not held_marker.exists():
            assert time.time() < deadline and other.poll() is None, 'the other process never took the slot'
            time.sleep(0.05)
        slot = FileTrainingSlot(lock_path)
        assert not slot.try_acquire('me')
        assert slot.holder() == 'other process'
        other.kill()
        other.wait()
        assert slot.try_acquire('me')
    finally:
        other.kill()


def test_without_the_variable_the_slot_is_always_free(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    slot = training_slot_from_env()
    assert isinstance(slot, AlwaysFreeSlot)
    assert slot.holder() is None
    assert slot.try_acquire('me')
    assert slot.holder() == 'me'


def test_the_variable_selects_the_file(monkeypatch: pytest.MonkeyPatch, lock_path: Path):
    monkeypatch.setenv(ENV_VAR, str(lock_path))
    slot = training_slot_from_env()
    assert isinstance(slot, FileTrainingSlot)
    assert slot.path == lock_path


def test_a_missing_directory_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv(ENV_VAR, str(tmp_path / 'not_mounted' / 'training.lock'))
    with pytest.raises(FileNotFoundError):
        training_slot_from_env()


class _Trainer(TrainerLogicGeneric):

    @property
    def training_progress(self) -> float | None:
        return None

    @property
    def detection_progress(self) -> float | None:
        return None

    @property
    def model_architecture(self) -> str | None:
        return None

    @property
    def provided_pretrained_models(self) -> list[PretrainedModel]:
        return []

    async def _train(self) -> None:
        pass

    async def _do_detections(self) -> None:
        pass

    def _get_new_best_training_state(self) -> TrainingStateData | None:
        return None

    def _on_metrics_published(self, training_state_data: TrainingStateData) -> None:
        pass

    async def _get_latest_model_files(self) -> dict[str, list[str]]:
        return {}

    async def _clear_training_data(self, training_folder: str) -> None:
        pass


def _trainer_with(slot: FileTrainingSlot) -> _Trainer:
    trainer = _Trainer('mocked', training_slot=slot)
    trainer._node = SimpleNamespace(uuid='node-uuid', name='node')  # type: ignore[assignment]
    return trainer


def test_an_idle_trainer_reports_blocked_while_a_sibling_holds_the_slot(lock_path: Path):
    sibling = FileTrainingSlot(lock_path)
    trainer = _trainer_with(FileTrainingSlot(lock_path))
    assert trainer.state == 'idle'
    sibling.try_acquire('sibling')
    assert trainer.state == 'blocked'
    sibling.release()
    assert trainer.state == 'idle'


async def test_a_training_waits_for_the_slot_and_reports_waiting_meanwhile(lock_path: Path):
    sibling = FileTrainingSlot(lock_path)
    sibling.try_acquire('sibling')
    trainer = _trainer_with(FileTrainingSlot(lock_path))
    trainer._training = SimpleNamespace(id='training-id', training_state='initialized')  # type: ignore[assignment]
    trainer._active_training_io = object()  # type: ignore[assignment]

    task = asyncio.create_task(trainer._acquire_training_slot())
    await asyncio.sleep(0.3)
    assert not task.done()
    assert trainer.state == 'waiting_for_slot'

    sibling.release()
    started = time.time()
    await asyncio.wait_for(task, timeout=5)
    assert time.time() - started < 2
    assert trainer.training_slot.held
    assert trainer.state == 'initialized'
    assert sibling.holder() == f'pid {os.getpid()} on {socket.gethostname()}, node node-uuid, training training-id'


def test_a_blocked_trainer_runs_into_its_idle_timeout(lock_path: Path):
    sibling = FileTrainingSlot(lock_path)
    sibling.try_acquire('sibling')
    node = SimpleNamespace(trainer_logic=_trainer_with(FileTrainingSlot(lock_path)), log=logging.getLogger(__name__),
                           _idle_timeout=1.0, _first_idle_time=time.time() - 2.0)
    with pytest.raises(SystemExit):
        TrainerNode.check_idle_timeout(node)  # type: ignore[arg-type]
