import logging
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from dacite import from_dict
from fastapi.encoders import jsonable_encoder

from ...data_classes import Hyperparameter, TrainingStatus
from ...enums import TrainerState
from ...helpers.misc import ensure_socket_response
from ...trainer.batch_size import REQUESTED_BATCH_SIZE
from ...trainer.exceptions import CriticalError, HyperparameterMismatchError
from ...trainer.hyperparameters import check_declaration, parse_pairs, resolve, warn_about_undeclared
from ...trainer.trainer_logic_generic import TrainerLogicGeneric
from ...trainer.trainer_node import TrainerNode
from ...trainer.training_slot import AlwaysFreeSlot
from .test_training_slot import TRAINING_CONFIG, _Trainer, _with_node

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
    assert resolve(DECLARED, {'use_amp': '0.5'})['use_amp'] is True
    assert resolve(DECLARED, {'use_amp': '0.0'})['use_amp'] is False


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
    {'use_amp': 'nan'},
    {'comment': True},
    {'comment': {'a': 1}},
    {'comment': [1, 2]},
])
def test_an_invalid_value_is_a_critical_error(received: dict[str, Any]):
    with pytest.raises(CriticalError, match=next(iter(received))):
        resolve(DECLARED, received)


@pytest.mark.parametrize('declaration', [
    Hyperparameter(name='h', type='string', default=''),  # type: ignore[arg-type]
    Hyperparameter(name='h', type='str', default='', minimum=1),
    Hyperparameter(name='h', type='bool', default=False, maximum=1),
    Hyperparameter(name='h', type='int', default=1, choices=['1']),
    Hyperparameter(name='h', type='int', default=5, minimum=10, maximum=1),
    Hyperparameter(name='h', type='int', default=None, minimum='1'),  # type: ignore[arg-type]
    Hyperparameter(name='h', type='float', default=None, maximum=True),
    Hyperparameter(name='h', type='int', default=0, minimum=1),
    Hyperparameter(name='h', type='int', default='many'),
    Hyperparameter(name='h', type='str', default='x', choices=['a', 'b']),
    Hyperparameter(name='h', type='pairs', default='', choices=['a:b']),
    Hyperparameter(name='h', type='pairs', default='a'),
])
def test_a_declaration_that_contradicts_itself_is_rejected(declaration: Hyperparameter):
    with pytest.raises(ValueError, match="the trainer declares hyperparameter 'h'"):
        check_declaration([declaration])


def test_a_name_declared_twice_is_rejected():
    with pytest.raises(ValueError, match='twice: epochs'):
        check_declaration([*DECLARED, Hyperparameter(name='epochs', type='int', default=1)])


def test_a_consistent_declaration_passes():
    check_declaration(DECLARED)
    check_declaration([Hyperparameter(name='learning_rate', type='float', default=None, minimum=0)])


def test_a_default_of_none_is_left_to_the_trainer():
    declared = [Hyperparameter(name='learning_rate', type='float', default=None, minimum=0)]
    assert resolve(declared, {}) == {}
    assert resolve(declared, {'learning_rate': None}) == {}
    assert resolve(declared, {'learning_rate': ''}) == {}


def test_undeclared_names_are_kept():
    resolved = resolve(DECLARED, {'flip_rl': True, 'batch_size': 16})
    assert resolved['flip_rl'] is True
    assert resolved['batch_size'] == 16


def test_resolving_twice_changes_nothing():
    once = resolve(DECLARED, {'epochs': '7', 'use_amp': 0.5, 'optimizer': '', 'trainer_version': '1.0'})
    assert resolve(DECLARED, once) == once


PAIRS = [Hyperparameter(name='flip_label_pairs', type='pairs', default='')]


def test_pairs_are_written_the_same_way_whatever_spacing_they_were_sent_with():
    assert resolve(PAIRS, {'flip_label_pairs': ' a : b , c:d,'})['flip_label_pairs'] == 'a:b,c:d'
    assert parse_pairs('a:b,c:d') == [('a', 'b'), ('c', 'd')]


def test_no_pairs_is_a_value_of_its_own():
    declared = [Hyperparameter(name='flip_label_pairs', type='pairs', default='a:b')]
    assert resolve(declared, {'flip_label_pairs': ''})['flip_label_pairs'] == ''
    assert resolve(declared, {})['flip_label_pairs'] == 'a:b'
    assert parse_pairs('') == []


@pytest.mark.parametrize('value', ['a', 'a:b:c', ':b', 'a:', 'a:b,c', 7, True])
def test_a_malformed_pair_is_a_critical_error(value: Any):
    with pytest.raises(CriticalError, match='flip_label_pairs'):
        resolve(PAIRS, {'flip_label_pairs': value})


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

async def test_begin_training_resolves_the_hyperparameters(data_folder: Path, started: list[TrainerLogicGeneric]):
    trainer = _with_node(_DeclaringTrainer('mocked', training_slot=AlwaysFreeSlot()))
    await trainer.begin_training('zauberzeug', 'demo', _config({'epochs': '7'}))
    assert started == [trainer]
    assert trainer.training.hyperparameters == {**resolve(DECLARED, {}), 'epochs': 7}


async def test_begin_training_is_rejected_for_a_value_that_does_not_fit(data_folder: Path,
                                                                        started: list[TrainerLogicGeneric]):
    old_training = data_folder / 'zauberzeug' / 'demo' / 'trainings' / 'old'
    old_training.mkdir(parents=True)
    trainer = _with_node(_DeclaringTrainer('mocked', training_slot=AlwaysFreeSlot()))
    with pytest.raises(HyperparameterMismatchError, match='learning_rate'):
        await trainer.begin_training('zauberzeug', 'demo', _config({'learning_rate': 1.5}))
    assert not started
    assert not trainer.training_active
    assert not trainer.training_slot.held
    assert old_training.exists()


async def test_the_loop_receives_a_mismatch_as_a_failure_that_is_no_refusal(data_folder: Path,
                                                                            started: list[TrainerLogicGeneric]):
    handlers = {}
    sio_client = SimpleNamespace(event=lambda handler: handlers.setdefault(handler.__name__, handler))
    trainer = _with_node(_DeclaringTrainer('mocked', training_slot=AlwaysFreeSlot()))
    node = SimpleNamespace(trainer_logic=trainer, log=logging.getLogger(__name__))
    TrainerNode.register_sio_events(node, sio_client)  # type: ignore[arg-type]

    @ensure_socket_response
    async def trigger(event: str, *args):
        return await handlers[event](*args)

    response = await trigger('begin_training', 'zauberzeug', 'demo', _config({'epochs': 0}))
    assert response == {'success': False, 'error_msg': "hyperparameter 'epochs' must be at least 1, got 0",
                        'payload': None}
    assert not started


async def test_a_trainer_that_declares_nothing_gets_the_hyperparameters_unchecked(data_folder: Path,
                                                                                  started: list[TrainerLogicGeneric]):
    trainer = _with_node(_Trainer('mocked', training_slot=AlwaysFreeSlot()))
    await trainer.begin_training('zauberzeug', 'demo', _config({'epochs': 'many'}))
    assert trainer.training.hyperparameters == {'epochs': 'many'}


async def test_a_resumed_training_is_resolved_again(data_folder: Path, started: list[TrainerLogicGeneric]):
    await _saved_by_an_older_trainer({'epochs': '7', 'trainer_version': '1.0'})
    trainer = _with_node(_DeclaringTrainer('mocked', training_slot=AlwaysFreeSlot()))
    assert await trainer.try_continue_run_if_incomplete()
    assert trainer.training.hyperparameters == {**resolve(DECLARED, {}), 'epochs': 7, 'trainer_version': '1.0'}
    assert trainer.training.training_state == TrainerState.TrainingRunning


async def test_a_resumed_training_that_does_not_fit_continues_with_its_saved_values(
        data_folder: Path, started: list[TrainerLogicGeneric], caplog: pytest.LogCaptureFixture):
    await _saved_by_an_older_trainer({'learning_rate': 1.5})
    trainer = _with_node(_DeclaringTrainer('mocked', training_slot=AlwaysFreeSlot()))
    with caplog.at_level(logging.WARNING):
        assert await trainer.try_continue_run_if_incomplete()
    assert trainer.training.hyperparameters == {'learning_rate': 1.5}
    assert trainer.training.training_state == TrainerState.TrainingRunning
    assert not trainer.errors.errors
    assert 'learning_rate' in caplog.text


def test_a_trainer_node_is_not_built_around_a_declaration_that_contradicts_itself():
    with pytest.raises(ValueError, match="hyperparameter 'epochs'"):
        TrainerNode(name='trainer', trainer_logic=_BrokenTrainer('mocked', training_slot=AlwaysFreeSlot()))


class _DeclaringTrainer(_Trainer):

    @property
    def provided_hyperparameters(self) -> list[Hyperparameter]:
        return DECLARED


class _BrokenTrainer(_Trainer):

    @property
    def provided_hyperparameters(self) -> list[Hyperparameter]:
        return [Hyperparameter(name='epochs', type='int', default=0, minimum=1)]


async def _saved_by_an_older_trainer(hyperparameters: dict[str, Any]) -> None:
    """Leaves a training behind that was begun before its trainer declared hyperparameters."""
    trainer = _with_node(_Trainer('mocked', training_slot=AlwaysFreeSlot()))
    await trainer.begin_training('zauberzeug', 'demo', _config(hyperparameters))
    trainer.training.training_state = TrainerState.TrainingRunning
    trainer.last_training_io.save(trainer.training)


def _config(hyperparameters: dict[str, Any]) -> dict[str, Any]:
    return {**TRAINING_CONFIG, 'hyperparameters': hyperparameters}
