from dataclasses import asdict

import pytest
from dacite import from_dict
from fastapi.encoders import jsonable_encoder

from ...data_classes import Hyperparameter, TrainingStatus
from ...trainer.exceptions import CriticalError
from ...trainer.hyperparameters import resolve

DECLARED = [
    Hyperparameter(name='epochs', type='int', default=100, minimum=1),
    Hyperparameter(name='learning_rate', type='float', default=0.01, minimum=0.0, maximum=1.0, editable=False),
    Hyperparameter(name='use_amp', type='bool', default=False, expert=True),
    Hyperparameter(name='optimizer', type='str', default='sgd', choices=['sgd', 'adam']),
    Hyperparameter(name='comment', type='str', default='none'),
]


def test_unsent_hyperparameters_get_their_declared_default():
    assert resolve(DECLARED, {}) == {'epochs': 100, 'learning_rate': 0.01, 'use_amp': False,
                                     'optimizer': 'sgd', 'comment': 'none'}


def test_an_empty_value_counts_as_unsent_except_for_a_str():
    resolved = resolve(DECLARED, {'epochs': '', 'comment': ''})
    assert resolved['epochs'] == 100
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


@pytest.mark.parametrize('received', [
    {'epochs': 7.5},
    {'epochs': 'many'},
    {'epochs': True},
    {'epochs': 0},
    {'learning_rate': 1.5},
    {'learning_rate': 'nan'},
    {'use_amp': 'maybe'},
    {'optimizer': 'rmsprop'},
])
def test_an_invalid_value_is_a_critical_error(received):
    with pytest.raises(CriticalError, match=next(iter(received))):
        resolve(DECLARED, received)


def test_undeclared_names_are_kept():
    resolved = resolve(DECLARED, {'flip_rl': True, 'batch_size': 16})
    assert resolved['flip_rl'] is True
    assert resolved['batch_size'] == 16


def test_the_declaration_survives_the_trip_to_the_loop():
    status = TrainingStatus(id='1', name='trainer', state='idle', uptime=0, errors={}, progress=None,
                            hyperparameters=DECLARED)
    assert from_dict(data_class=TrainingStatus, data=jsonable_encoder(asdict(status))) == status
