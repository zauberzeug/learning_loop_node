from pathlib import Path

import pytest

from ...globals import GLOBALS
from ...trainer.trainer_logic_generic import TrainerLogicGeneric


@pytest.fixture
def data_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(GLOBALS, 'data_folder', str(tmp_path / 'data'))
    return tmp_path / 'data'


@pytest.fixture
def started(monkeypatch: pytest.MonkeyPatch) -> list[TrainerLogicGeneric]:
    """The trainers whose training task would have been started; nothing actually runs."""
    trainers: list[TrainerLogicGeneric] = []

    def record(trainer: TrainerLogicGeneric) -> None:
        trainers.append(trainer)
    monkeypatch.setattr(TrainerLogicGeneric, '_begin_training_task', record)
    return trainers
