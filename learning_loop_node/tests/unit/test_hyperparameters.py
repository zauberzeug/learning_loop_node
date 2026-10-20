import logging
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from dacite import from_dict
from fastapi.encoders import jsonable_encoder

from ...data_classes import Hyperparameter, TrainingStatus
from ...enums import TrainerState
from ...trainer.batch_size import REQUESTED_BATCH_SIZE
from ...trainer.exceptions import CriticalError, TrainingRefusedError
from ...trainer.hyperparameters import resolve, warn_about_undeclared
from ...trainer.io_helpers import LastTrainingIO
from ...trainer.training_slot import AlwaysFreeSlot
from .test_training_slot import TRAINING_CONFIG, _Trainer

# pylint: disable=protected-access,redefined-outer-name

DECLARED = [
    Hyperparameter(name='epochs', type='int', default=100, minimum=1),
    Hyperparameter(name='learning_rate', type='float', default=0.01, minimum=0.0, maximum=1.0),
    Hyperparameter(name='use_amp', type='bool', default=False, expert=True),
    Hyperparameter(name='optimizer', type='str', default='sgd', choices=['sgd', 'adam']),
    Hyperparameter(name='comment', type='str', default='none'),
]


def test_unsent_hyperparameters_get_their_declared_default():
    assert resolve(DECLARED, {}) == {'epochs': 100, 'learning_rate': 0.01, 'use_amp': False,
                                     'optimizer': 'sgd', 'comment': 'none'}


def test_an_empty_value_counts_as_unsent_except_for_a_free_text():
    resolved = resolve(DECLARED, {'epochs': '', 'optimizer': '', 'comment': ''})
    assert resolved['epochs'] == 100
    assert resolved['optimizer'] == 'sgd'
    assert resolved['comment'] == ''


def test_sent_values_are_coerced_to_their_declared_type():
    resolved = resolve(DECLARED, {'epochs': '7', 'learning_rate': '0.5', 'use_amp': 'true', 'optimizer': 'adam'})
    assert resolved['epochs'] == 7 and isinstance(resolved['epochs'], int)
    assert resolved['learning_rate'] == 0.5
    assert resolved['use_amp'] is True
    assert resolved['optimizer'] == 'adam'


def test_a_number_is_accepted_for_a_bool():
    assert resolve(DECLARED, {'use_amp': 0.5})['use_amp'] is True
    assert resolve(DECLARED, {'use_amp': 0})['use_amp'] is False


def test_a_whole_float_is_accepted_for_an_int():
    assert resolve(DECLARED, {'epochs': 7.0})['epochs'] == 7
    assert resolve(DECLARED, {'epochs': '7.0'})['epochs'] == 7


def test_a_large_int_is_kept_exact():
    assert resolve(DECLARED, {'epochs': '12345678901234567891'})['epochs'] == 12345678901234567891


@pytest.mark.parametrize('received', [
    {'epochs': 7.5},
    {'epochs': 'many'},
    {'epochs': True},
    {'epochs': 0},
    {'learning_rate': 1.5},
    {'learning_rate': 'nan'},
    {'use_amp': 'maybe'},
    {'use_amp': float('nan')},
    {'optimizer': 'rmsprop'},
    {'comment': {'a': 1}},
    {'comment': [1, 2]},
])
def test_an_invalid_value_is_a_critical_error(received):
    with pytest.raises(CriticalError, match=next(iter(received))):
        resolve(DECLARED, received)


@pytest.mark.parametrize('declaration', [
    Hyperparameter(name='h', type='string', default=''),  # type: ignore[arg-type]
    Hyperparameter(name='h', type='str', default='', minimum=1),
    Hyperparameter(name='h', type='bool', default=False, maximum=1),
    Hyperparameter(name='h', type='int', default=1, choices=['1']),
    Hyperparameter(name='h', type='int', default=5, minimum=10, maximum=1),
    Hyperparameter(name='h', type='int', default=0, minimum=1),
    Hyperparameter(name='h', type='int', default='many'),
    Hyperparameter(name='h', type='str', default='x', choices=['a', 'b']),
])
@pytest.mark.parametrize('received', [{}, {'h': '5'}])
def test_a_declaration_that_contradicts_itself_is_a_critical_error(declaration, received):
    with pytest.raises(CriticalError, match="the trainer declares hyperparameter 'h'"):
        resolve([declaration], received)


def test_a_default_of_none_is_left_to_the_trainer():
    declared = [Hyperparameter(name='learning_rate', type='float', default=None, minimum=0)]
    assert resolve(declared, {}) == {'learning_rate': None}


def test_undeclared_names_are_kept():
    resolved = resolve(DECLARED, {'flip_rl': True, 'batch_size': 16})
    assert resolved['flip_rl'] is True
    assert resolved['batch_size'] == 16


def test_resolving_twice_changes_nothing():
    once = resolve(DECLARED, {'epochs': '7', 'use_amp': 0.5, 'optimizer': '', 'trainer_version': '1.0'})
    assert resolve(DECLARED, once) == once


def test_only_names_neither_the_trainer_nor_the_library_knows_are_warned_about(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING):
        warn_about_undeclared(DECLARED, {'epochs': 7, REQUESTED_BATCH_SIZE: 16, 'batch_size': 16})
    assert 'batch_size' in caplog.text
    assert REQUESTED_BATCH_SIZE not in caplog.text


def test_the_declaration_survives_the_trip_to_the_loop():
    status = TrainingStatus(id='1', name='trainer', state='idle', uptime=0, errors={}, progress=None,
                            hyperparameters=DECLARED)
    assert from_dict(data_class=TrainingStatus, data=jsonable_encoder(asdict(status))) == status


# --- resolving in the trainer ---

class _DeclaringTrainer(_Trainer):

    @property
    def provided_hyperparameters(self) -> list[Hyperparameter]:
        return DECLARED


def _with_node(trainer: _Trainer) -> _Trainer:
    trainer._node = SimpleNamespace(uuid='node-uuid', name='node', loop_communicator=None)  # type: ignore[assignment]
    trainer._last_training_io = LastTrainingIO('node-uuid')
    return trainer


def _config(hyperparameters: dict) -> dict:
    return {**TRAINING_CONFIG, 'hyperparameters': hyperparameters}


async def test_begin_training_resolves_the_hyperparameters(data_folder: Path, started: list):
    trainer = _with_node(_DeclaringTrainer('mocked', training_slot=AlwaysFreeSlot()))
    await trainer.begin_training('zauberzeug', 'demo', _config({'epochs': '7'}))
    assert started == [trainer]
    assert trainer.training.hyperparameters == {**resolve(DECLARED, {}), 'epochs': 7}


async def test_begin_training_is_refused_for_a_value_that_does_not_fit(data_folder: Path, started: list):
    old_training = data_folder / 'zauberzeug' / 'demo' / 'trainings' / 'old'
    old_training.mkdir(parents=True)
    trainer = _with_node(_DeclaringTrainer('mocked', training_slot=AlwaysFreeSlot()))
    with pytest.raises(TrainingRefusedError, match='learning_rate'):
        await trainer.begin_training('zauberzeug', 'demo', _config({'learning_rate': 1.5}))
    assert not started
    assert not trainer.training_active
    assert not trainer.training_slot.held
    assert old_training.exists()


async def test_a_trainer_that_declares_nothing_gets_the_hyperparameters_unchecked(data_folder: Path, started: list):
    trainer = _with_node(_Trainer('mocked', training_slot=AlwaysFreeSlot()))
    await trainer.begin_training('zauberzeug', 'demo', _config({'epochs': 'many'}))
    assert trainer.training.hyperparameters == {'epochs': 'many'}


async def _saved_by_an_older_trainer(hyperparameters: dict) -> None:
    """Leaves a training behind that was begun before its trainer declared hyperparameters."""
    trainer = _with_node(_Trainer('mocked', training_slot=AlwaysFreeSlot()))
    await trainer.begin_training('zauberzeug', 'demo', _config(hyperparameters))
    trainer.training.training_state = TrainerState.TrainingRunning
    trainer.last_training_io.save(trainer.training)


async def test_a_resumed_training_is_resolved_again(data_folder: Path, started: list):
    await _saved_by_an_older_trainer({'epochs': '7', 'trainer_version': '1.0'})
    trainer = _with_node(_DeclaringTrainer('mocked', training_slot=AlwaysFreeSlot()))
    assert await trainer.try_continue_run_if_incomplete()
    assert trainer.training.hyperparameters == {**resolve(DECLARED, {}), 'epochs': 7, 'trainer_version': '1.0'}
    assert trainer.training.training_state == TrainerState.TrainingRunning


async def test_a_resumed_training_that_does_not_fit_is_cleaned_up_with_an_error(data_folder: Path, started: list):
    await _saved_by_an_older_trainer({'learning_rate': 1.5})
    trainer = _with_node(_DeclaringTrainer('mocked', training_slot=AlwaysFreeSlot()))
    assert await trainer.try_continue_run_if_incomplete()
    assert trainer.training.training_state == TrainerState.ReadyForCleanup
    assert 'learning_rate' in trainer.errors.errors['prepare']
