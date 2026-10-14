import asyncio
import logging
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ...data_classes import Context, PretrainedModel, TrainingStateData
from ...globals import GLOBALS
from ...helpers.misc import ensure_socket_response
from ...trainer.exceptions import TrainingRefusedError, TrainingSlotTakenError
from ...trainer.io_helpers import LastTrainingIO
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


def _holder_of(lock_path: Path) -> str | None:
    return FileTrainingSlot(lock_path)._sibling_holder()  # pylint: disable=protected-access


def _claimed(training: str) -> str:
    return f'pid {os.getpid()} on {socket.gethostname()}, {training}'


def test_a_free_slot_is_not_taken(lock_path: Path):
    assert not FileTrainingSlot(lock_path).taken_by_sibling
    assert _holder_of(lock_path) is None


def test_the_slot_is_exclusive_and_names_its_holder(lock_path: Path):
    first, second = FileTrainingSlot(lock_path), FileTrainingSlot(lock_path)
    first.claim('node a')
    assert first.held
    with pytest.raises(TrainingSlotTakenError):
        second.claim('node b')
    assert not second.held
    assert second.taken_by_sibling
    assert not first.taken_by_sibling
    assert _holder_of(lock_path) == _claimed('node a')


def test_inspecting_a_slot_does_not_create_its_file(lock_path: Path):
    assert not FileTrainingSlot(lock_path).taken_by_sibling
    assert not lock_path.exists()


def test_the_file_is_writable_for_sibling_users(lock_path: Path):
    slot = FileTrainingSlot(lock_path)
    old_umask = os.umask(0o022)
    try:
        slot.claim('me')
    finally:
        os.umask(old_umask)
        slot.release()
    assert lock_path.stat().st_mode & 0o777 == 0o666


@pytest.mark.skipif(os.geteuid() == 0, reason='root reads any file')
def test_an_unreadable_slot_counts_as_free(lock_path: Path):
    lock_path.touch(mode=0o000)
    assert not FileTrainingSlot(lock_path).taken_by_sibling


def test_release_hands_the_slot_on(lock_path: Path):
    first, second = FileTrainingSlot(lock_path), FileTrainingSlot(lock_path)
    first.claim('node a')
    first.release()
    assert not first.held
    second.claim('node b')
    assert first.taken_by_sibling
    assert _holder_of(lock_path) == _claimed('node b')


def test_acquiring_twice_is_harmless(lock_path: Path):
    slot = FileTrainingSlot(lock_path)
    slot.claim('node a')
    slot.claim('node a')
    slot.release()
    slot.release()
    assert not slot.held
    assert _holder_of(lock_path) is None


def _failing(*args):
    raise OSError(28, 'No space left on device')


def test_a_failed_claim_leaves_the_slot_free(lock_path: Path, monkeypatch: pytest.MonkeyPatch):
    slot = FileTrainingSlot(lock_path)
    with monkeypatch.context() as patch:
        patch.setattr(os, 'write', _failing)
        with pytest.raises(OSError):
            slot.claim('me')
    assert not slot.held
    FileTrainingSlot(lock_path).claim('sibling')


def test_a_failed_release_still_frees_the_slot(lock_path: Path, monkeypatch: pytest.MonkeyPatch):
    slot = FileTrainingSlot(lock_path)
    slot.claim('me')
    with monkeypatch.context() as patch:
        patch.setattr(os, 'ftruncate', _failing)
        slot.release()
    assert not slot.held
    FileTrainingSlot(lock_path).claim('sibling')


def test_a_dying_process_frees_the_slot(lock_path: Path):
    held_marker = lock_path.with_suffix('.held')
    other = subprocess.Popen([sys.executable, '-c', f'''
from learning_loop_node.trainer.training_slot import FileTrainingSlot
import pathlib, time
FileTrainingSlot({str(lock_path)!r}).claim('other process')
pathlib.Path({str(held_marker)!r}).touch()
time.sleep(60)
'''], stdout=subprocess.DEVNULL)
    try:
        deadline = time.time() + 30
        while not held_marker.exists():
            assert time.time() < deadline and other.poll() is None, 'the other process never took the slot'
            time.sleep(0.05)
        slot = FileTrainingSlot(lock_path)
        with pytest.raises(TrainingSlotTakenError):
            slot.claim('me')
        assert slot.taken_by_sibling
        assert _holder_of(lock_path).endswith(', other process')
        other.kill()
        other.wait()
        slot.claim('me')
    finally:
        other.kill()


def test_a_forked_child_does_not_keep_the_slot(lock_path: Path):
    slot = FileTrainingSlot(lock_path)
    slot.claim('me')
    started, signal = os.pipe()
    child = os.fork()
    if child == 0:
        os.write(signal, b'x')
        time.sleep(30)
        os._exit(0)
    try:
        os.read(started, 1)
        slot.release()
        FileTrainingSlot(lock_path).claim('sibling')
    finally:
        os.kill(child, 9)
        os.waitpid(child, 0)
        os.close(started)
        os.close(signal)


def test_without_the_variable_the_slot_is_always_free(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    slot = training_slot_from_env()
    assert isinstance(slot, AlwaysFreeSlot)
    assert not slot.taken_by_sibling
    slot.claim('me')
    assert slot.held
    slot.release()
    assert not slot.held


class _ProbedSlot(AlwaysFreeSlot):
    """Loses its first attempts against a sibling's probe, or against a sibling that holds the slot."""

    def __init__(self, failed_attempts: int, sibling: str | None) -> None:
        super().__init__()
        self.failed_attempts = failed_attempts
        self.sibling = sibling

    def _acquire(self, holder: str) -> None:
        if self.failed_attempts:
            self.failed_attempts -= 1
            raise TrainingSlotTakenError(self.sibling or 'unknown')
        super()._acquire(holder)

    def _sibling_holder(self) -> str | None:
        return self.sibling


def test_a_claim_tries_again_after_losing_against_a_probe():
    slot = _ProbedSlot(failed_attempts=2, sibling=None)
    slot.claim('me')
    assert slot.held


def test_a_claim_gives_up_on_a_slot_that_stays_locked():
    slot = _ProbedSlot(failed_attempts=10, sibling=None)
    with pytest.raises(TrainingSlotTakenError):
        slot.claim('me')
    assert not slot.held


def test_a_claim_is_refused_at_once_while_a_sibling_holds_the_slot():
    slot = _ProbedSlot(failed_attempts=10, sibling='sibling')
    with pytest.raises(TrainingSlotTakenError, match='held by sibling'):
        slot.claim('me')
    assert slot.failed_attempts == 9


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
    trainer._node = SimpleNamespace(uuid='node-uuid', name='node', loop_communicator=None)  # type: ignore[assignment]
    return trainer


TRAINING_CONFIG = {'model_variant': 'small', 'categories': [], 'training_number': 7, 'hyperparameters': {}}


@pytest.fixture
def data_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(GLOBALS, 'data_folder', str(tmp_path / 'data'))
    return tmp_path / 'data'


@pytest.fixture
def started(monkeypatch: pytest.MonkeyPatch) -> list[TrainerLogicGeneric]:
    trainers: list[TrainerLogicGeneric] = []

    def record(trainer: TrainerLogicGeneric) -> None:
        trainers.append(trainer)
    monkeypatch.setattr(TrainerLogicGeneric, '_begin_training_task', record)
    return trainers


def test_an_idle_trainer_reports_blocked_while_a_sibling_holds_the_slot(lock_path: Path):
    sibling = FileTrainingSlot(lock_path)
    trainer = _trainer_with(FileTrainingSlot(lock_path))
    assert trainer.state == 'idle'
    sibling.claim('sibling')
    assert trainer.state == 'blocked'
    sibling.release()
    assert trainer.state == 'idle'


async def test_begin_training_takes_the_slot_and_starts(lock_path: Path, data_folder: Path, started: list):
    trainer = _trainer_with(FileTrainingSlot(lock_path))
    await trainer.begin_training('zauberzeug', 'demo', TRAINING_CONFIG)
    assert started == [trainer]
    assert trainer.training_slot.held
    assert _holder_of(lock_path) == _claimed('node node-uuid, training #7 of zauberzeug/demo')


async def test_begin_training_is_refused_while_a_sibling_holds_the_slot(lock_path: Path, data_folder: Path,
                                                                        started: list):
    old_training = data_folder / 'zauberzeug' / 'demo' / 'trainings' / 'old'
    old_training.mkdir(parents=True)
    sibling = FileTrainingSlot(lock_path)
    sibling.claim('sibling')
    trainer = _trainer_with(FileTrainingSlot(lock_path))

    with pytest.raises(TrainingRefusedError, match=re.escape(_claimed('sibling'))):
        await trainer.begin_training('zauberzeug', 'demo', TRAINING_CONFIG)
    assert not started
    assert not trainer.training_active
    assert not trainer.training_slot.held
    assert old_training.exists()
    assert trainer.state == 'blocked'


async def test_begin_training_is_refused_while_a_training_is_active(lock_path: Path, data_folder: Path,
                                                                     started: list):
    trainer = _trainer_with(FileTrainingSlot(lock_path))
    await trainer.begin_training('zauberzeug', 'demo', TRAINING_CONFIG)
    with pytest.raises(TrainingRefusedError, match='still active'):
        await trainer.begin_training('zauberzeug', 'demo', TRAINING_CONFIG)
    assert started == [trainer]


async def test_an_invalid_training_config_gives_the_slot_back(lock_path: Path, data_folder: Path, started: list):
    trainer = _trainer_with(FileTrainingSlot(lock_path))
    with pytest.raises(ValueError):
        await trainer.begin_training('zauberzeug', 'demo', {**TRAINING_CONFIG, 'training_number': 'seven'})
    assert not started
    assert not trainer.training_slot.held
    assert _holder_of(lock_path) is None


async def test_the_loop_receives_a_refusal_as_a_failed_socket_response(lock_path: Path, data_folder: Path,
                                                                       started: list):
    sibling = FileTrainingSlot(lock_path)
    sibling.claim('sibling')
    handlers = {}
    sio_client = SimpleNamespace(event=lambda handler: handlers.setdefault(handler.__name__, handler))
    node = SimpleNamespace(trainer_logic=_trainer_with(FileTrainingSlot(lock_path)), log=logging.getLogger(__name__))
    TrainerNode.register_sio_events(node, sio_client)  # type: ignore[arg-type]

    @ensure_socket_response
    async def trigger(event: str, *args):
        return await handlers[event](*args)

    response = await trigger('begin_training', 'zauberzeug', 'demo', TRAINING_CONFIG)
    assert response == {'success': False, 'error_msg': f'the training slot is held by {_claimed("sibling")}',
                        'payload': None}

    sibling.release()
    response = await trigger('begin_training', 'zauberzeug', 'demo', TRAINING_CONFIG)
    assert response == {'success': True, 'error_msg': None, 'payload': None}


class _TrainerFailingCleanup(_Trainer):

    async def _clear_training_data(self, training_folder: str) -> None:
        raise OSError('disk gone')


async def test_a_training_that_fails_to_clean_up_does_not_wedge_the_node(lock_path: Path, data_folder: Path,
                                                                          started: list):
    trainer = _TrainerFailingCleanup('mocked', training_slot=FileTrainingSlot(lock_path))
    trainer._node = SimpleNamespace(uuid='node-uuid', name='node', loop_communicator=None)  # type: ignore[assignment]
    trainer._last_training_io = LastTrainingIO('node-uuid')
    await trainer.begin_training('zauberzeug', 'demo', TRAINING_CONFIG)
    trainer.training.training_state = 'ready_for_cleanup'

    await trainer._run()

    assert not trainer.training_slot.held
    assert trainer.state == 'idle'
    await trainer.begin_training('zauberzeug', 'demo', TRAINING_CONFIG)
    assert len(started) == 2


async def test_a_resumed_training_waits_for_the_slot_and_reports_waiting_meanwhile(lock_path: Path):
    sibling = FileTrainingSlot(lock_path)
    sibling.claim('sibling')
    trainer = _trainer_with(FileTrainingSlot(lock_path))
    trainer._training = SimpleNamespace(id='training-id', training_state='initialized', training_number=3,  # type: ignore[assignment]
                                        context=Context(organization='zauberzeug', project='demo'))
    trainer._active_training_io = object()  # type: ignore[assignment]

    task = asyncio.create_task(trainer.training_slot.wait_for('training #3'))
    await asyncio.sleep(0.3)
    assert not task.done()
    assert trainer.state == 'waiting_for_slot'

    sibling.release()
    started = time.time()
    await asyncio.wait_for(task, timeout=5)
    assert time.time() - started < 2
    assert trainer.training_slot.held
    assert trainer.state == 'initialized'
    assert _holder_of(lock_path) == _claimed('training #3')


def test_a_blocked_trainer_runs_into_its_idle_timeout(lock_path: Path):
    sibling = FileTrainingSlot(lock_path)
    sibling.claim('sibling')
    node = SimpleNamespace(trainer_logic=_trainer_with(FileTrainingSlot(lock_path)), log=logging.getLogger(__name__),
                           _idle_timeout=1.0, _first_idle_time=time.time() - 2.0)
    with pytest.raises(SystemExit):
        TrainerNode.check_idle_timeout(node)  # type: ignore[arg-type]
