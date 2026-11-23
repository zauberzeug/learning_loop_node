from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ...data_classes import Context
from ...globals import GLOBALS
from ...trainer.trainer_node import TrainerNode
from ..trainer.testing_trainer_logic import TestingTrainerLogic

# pylint: disable=protected-access


async def test_a_training_persisted_in_an_unknown_state_is_cleaned_up(trainer: TestingTrainerLogic):
    trainer.training.training_state = 'train_model_downloaded'
    trainer.node.last_training_io.save(trainer.training)
    train = AsyncMock()
    trainer._train = train  # type: ignore[method-assign]

    await trainer._training_loop()

    assert trainer._training is None
    assert not trainer.node.last_training_io.exists()
    train.assert_not_awaited()


@pytest.fixture(name='trainer')
async def trainer_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[TestingTrainerLogic]:
    """A trainer with a fresh training and a node that reports its status nowhere."""
    monkeypatch.setattr(GLOBALS, 'data_folder', str(tmp_path))
    monkeypatch.delenv('RESTART_AFTER_TRAINING', raising=False)
    trainer = TestingTrainerLogic()
    node = TrainerNode(name='test', trainer_logic=trainer, uuid='NODE-000-0000-0000-0000-000000000000')
    monkeypatch.setattr(node, 'send_status', AsyncMock())
    trainer._init_new_training(Context(organization='zauberzeug', project='demo'),
                               {'categories': [], 'training_number': 0, 'model_variant': '', 'hyperparameters': {}})
    yield trainer
    await node.loop_communicator.shutdown()
