"""Resolving the hyperparameters a training was sent against the ones its trainer declares.

The loop sends only what the project sets, plus what the job overrides. Everything else comes from
the trainer's declared defaults, so after :func:`resolve` a training holds the full set it runs
with, and that is what gets reported back to the loop.
"""

import contextlib
import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

from ..data_classes import Hyperparameter
from .batch_size import REQUESTED_BATCH_SIZE
from .exceptions import CriticalError

logger = logging.getLogger(__name__)

LIBRARY_NAMES = (REQUESTED_BATCH_SIZE,)
"""Hyperparameters the library reads itself, so a trainer that does not declare them still uses them."""

BOOL_STRINGS = {'true': True, '1': True, 'yes': True, 'on': True,
                'false': False, '0': False, 'no': False, 'off': False}

TEXT_TYPES = ('str', 'pairs')
"""Types for which an empty value is a value of its own rather than an unset one."""


def resolve(declared: Sequence[Hyperparameter], received: Mapping[str, Any]) -> dict[str, Any]:
    """Coerce and check ``received`` against ``declared`` and fill in the declared defaults.

    ``declared`` is trusted to have passed :func:`check_declaration`. Names the trainer does not declare are kept
    unchanged: old project configurations still carry them, and a resolved training carries the values its trainer
    reported. An unset hyperparameter with a default of ``None`` is left out, so the trainer's own fallback applies.
    Resolving twice gives the same result.

    :raises CriticalError: If a value cannot be coerced to its type, is out of range or not one of its choices.
    """
    resolved = dict(received)
    for hyperparameter in declared:
        value = received.get(hyperparameter.name)
        if value is None or (value == '' and (hyperparameter.type not in TEXT_TYPES or hyperparameter.choices)):
            if hyperparameter.default is None:
                resolved.pop(hyperparameter.name, None)
            else:
                resolved[hyperparameter.name] = _coerce(hyperparameter, hyperparameter.default)
        else:
            resolved[hyperparameter.name] = _check(hyperparameter, _coerce(hyperparameter, value))
    return resolved


def check_declaration(declared: Sequence[Hyperparameter]) -> None:
    """Check that ``declared`` is consistent, so that :func:`resolve` can rely on it.

    A trainer's declaration does not change while it runs, so the node checks it once, when it is built. A default of
    ``None`` stands for "the trainer decides" and is not checked.

    :raises ValueError: If a name is declared twice, a hyperparameter contradicts itself or its default does not fit.
    """
    names = [h.name for h in declared]
    if duplicates := sorted({name for name in names if names.count(name) > 1}):
        raise ValueError(f'the trainer declares these hyperparameters twice: {", ".join(duplicates)}')
    for hyperparameter in declared:
        _check_one_declaration(hyperparameter)


def parse_pairs(text: str) -> list[tuple[str, str]]:
    """The ``key:value`` items of a ``pairs`` hyperparameter such as ``"uuid:uuid,uuid:uuid"``.

    Empty items and the spaces around keys and values are skipped, so ``''`` holds no pairs.

    :raises ValueError: If an item is not a key and a value separated by a single colon.
    """
    pairs = []
    for item in (i.strip() for i in text.split(',')):
        if not item:
            continue
        key, _, value = (part.strip() for part in item.partition(':'))
        if not key or not value or ':' in value:
            raise ValueError(f'"{item}" is not of the form "key:value"')
        pairs.append((key, value))
    return pairs


def warn_about_undeclared(declared: Sequence[Hyperparameter], received: Mapping[str, Any]) -> None:
    """Log the names in ``received`` that neither the trainer nor the library knows; they are not rejected."""
    unknown = sorted(set(received) - {h.name for h in declared} - set(LIBRARY_NAMES))
    if unknown:
        logger.warning('these hyperparameters are not declared by this trainer: %s', ', '.join(unknown))


def _check_one_declaration(hyperparameter: Hyperparameter) -> None:
    def contradiction(reason: str) -> ValueError:
        return ValueError(f"the trainer declares hyperparameter '{hyperparameter.name}' {reason}")

    if hyperparameter.type not in ('int', 'float', 'bool', 'str', 'pairs'):
        raise contradiction(f'with the unknown type {hyperparameter.type!r}')
    if hyperparameter.type not in ('int', 'float') and (hyperparameter.minimum is not None or
                                                        hyperparameter.maximum is not None):
        raise contradiction(f'as {hyperparameter.type} with a minimum or maximum')
    if hyperparameter.type != 'str' and hyperparameter.choices:
        raise contradiction(f'as {hyperparameter.type} with choices')
    for bound in (hyperparameter.minimum, hyperparameter.maximum):
        if bound is not None and (isinstance(bound, bool) or not isinstance(bound, (int, float))):
            raise contradiction(f'with the bound {bound!r}, which is not a number')
    if (hyperparameter.minimum is not None and hyperparameter.maximum is not None and
            hyperparameter.minimum > hyperparameter.maximum):
        raise contradiction('with a minimum above its maximum')
    if hyperparameter.default is not None:
        try:
            _check(hyperparameter, _coerce(hyperparameter, hyperparameter.default))
        except CriticalError as e:
            raise contradiction(f'with a default that does not fit: {e}') from None


def _coerce(hyperparameter: Hyperparameter, value: Any) -> Any:
    if hyperparameter.type == 'pairs':
        if not isinstance(value, str):
            raise CriticalError(f"hyperparameter '{hyperparameter.name}' must be text of key:value pairs, got {value!r}")
        try:
            return ','.join(f'{key}:{item}' for key, item in parse_pairs(value))
        except ValueError as e:
            raise CriticalError(f"hyperparameter '{hyperparameter.name}': {e}") from None
    if hyperparameter.type == 'str':
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise CriticalError(f"hyperparameter '{hyperparameter.name}' must be text, got {value!r}")
        return str(value)
    if hyperparameter.type == 'bool':
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in BOOL_STRINGS:
            return BOOL_STRINGS[value.strip().lower()]
        with contextlib.suppress(TypeError, ValueError):
            number = float(value)
            if math.isfinite(number):
                return number != 0
        raise CriticalError(f"hyperparameter '{hyperparameter.name}' must be true or false, got {value!r}")
    if isinstance(value, bool):
        raise CriticalError(f"hyperparameter '{hyperparameter.name}' must be a number, got {value!r}")
    if hyperparameter.type == 'int' and isinstance(value, (int, str)):
        with contextlib.suppress(ValueError):
            return int(value)
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
