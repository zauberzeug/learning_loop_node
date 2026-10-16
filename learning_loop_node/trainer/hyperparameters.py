"""Resolving the hyperparameters a training was sent against the ones its trainer declares.

The loop sends only what the project sets, plus what the job overrides. Everything else comes from
the trainer's declared defaults, so after :func:`resolve` a training holds the full set it runs
with, and that is what gets reported back to the loop.
"""

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

from ..data_classes import Hyperparameter
from .exceptions import CriticalError

logger = logging.getLogger(__name__)

REPORTED_NAMES = ('batch_size', 'trainer_version')
"""Keys a trainer writes into the hyperparameters to report them; they are never unrecognised input."""

BOOL_STRINGS = {'true': True, '1': True, 'yes': True, 'on': True,
                'false': False, '0': False, 'no': False, 'off': False}


def resolve(declared: Sequence[Hyperparameter], received: Mapping[str, Any]) -> dict[str, Any]:
    """Coerce and check ``received`` against ``declared`` and fill in the declared defaults.

    Names the trainer does not declare are kept and logged rather than rejected: old project
    configurations still carry them, and refusing them would refuse trainings that work today.

    :raises CriticalError: If a value cannot be coerced to its type, is out of range or not one of its choices.
    """
    resolved = dict(received)
    for hyperparameter in declared:
        value = received.get(hyperparameter.name)
        if value is None or (value == '' and hyperparameter.type != 'str'):
            resolved[hyperparameter.name] = hyperparameter.default
        else:
            resolved[hyperparameter.name] = _check(hyperparameter, _coerce(hyperparameter, value))

    unknown = sorted(set(received) - {h.name for h in declared} - set(REPORTED_NAMES))
    if unknown:
        logger.warning('these hyperparameters are not declared by this trainer: %s', ', '.join(unknown))
    return resolved


def _coerce(hyperparameter: Hyperparameter, value: Any) -> Any:
    if hyperparameter.type == 'str':
        return str(value)
    if hyperparameter.type == 'bool':
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0  # older projects configure a flag as a probability, e.g. fliplr: 0.5
        if str(value).strip().lower() in BOOL_STRINGS:
            return BOOL_STRINGS[str(value).strip().lower()]
        raise CriticalError(f"hyperparameter '{hyperparameter.name}' must be true or false, got {value!r}")
    if isinstance(value, bool):
        raise CriticalError(f"hyperparameter '{hyperparameter.name}' must be a number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise CriticalError(f"hyperparameter '{hyperparameter.name}' must be a number, got {value!r}") from None
    if not math.isfinite(number):
        raise CriticalError(f"hyperparameter '{hyperparameter.name}' must be finite, got {value!r}")
    if hyperparameter.type == 'float':
        return number
    if not number.is_integer():
        raise CriticalError(f"hyperparameter '{hyperparameter.name}' must be a whole number, got {value!r}")
    return int(number)


def _check(hyperparameter: Hyperparameter, value: Any) -> Any:
    if hyperparameter.minimum is not None and value < hyperparameter.minimum:
        raise CriticalError(f"hyperparameter '{hyperparameter.name}' must be at least {hyperparameter.minimum}, "
                            f'got {value}')
    if hyperparameter.maximum is not None and value > hyperparameter.maximum:
        raise CriticalError(f"hyperparameter '{hyperparameter.name}' must be at most {hyperparameter.maximum}, "
                            f'got {value}')
    if hyperparameter.choices and value not in hyperparameter.choices:
        raise CriticalError(f"hyperparameter '{hyperparameter.name}' must be one of {', '.join(hyperparameter.choices)}, "
                            f'got {value!r}')
    return value
