"""Declaring what a trainer accepts, instead of parsing it again at every use site.

The loop delivers a training's hyperparameters as a free-form ``dict[str, Any]`` whose values
arrive as strings, so each trainer grew its own coercion: ``int(hp.get('epochs', 130))`` in one
node, a yaml round-trip in another. Three consequences followed, and all three were found in the
field rather than in review:

* the same knob acquired a different default per node, so one loop configuration meant two
  different trainings;
* a knob a node did not happen to read was dropped without a word;
* a malformed value raised deep inside the run, once the GPU was already busy.

A schema addresses the class rather than the instances. A trainer declares its knobs once;
:func:`validate` coerces and range-checks them all *before* the run starts, names what it did not
recognise, and returns plain values.

The declarations are deliberately plain data — a name, a kind, a default, bounds — with no
callables anywhere, so the same list can later be reported to the loop and drive its
configuration UI. Keep it that way: a lambda in here is cheap today and un-serialisable later.
"""

import logging
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .exceptions import CriticalError

logger = logging.getLogger(__name__)

DETECT_NMS_CONF_THRES = 0.2
"""Minimum confidence for a detection to count, shared by every detector and validation pass."""

DETECT_NMS_IOU_THRES = 0.45
"""Maximum IoU between two detections of one category before the weaker one is suppressed."""

REPORTED_NAMES = ('batch_size', 'trainer_version')
"""Keys a trainer writes back into the training's hyperparameters to report them to the loop.

They travel in the same dict as the configured knobs, so they are never "unrecognised input"."""

FLIP_ALIASES = {'flip_rl': 'fliplr', 'flip_lr': 'fliplr', 'flip_ud': 'flipud'}
"""The loop's flip knob names mapped onto the yaml names a yolov5-style trainer expects.

``flip_rl`` is the original misspelling. It stays because projects are configured with it, and
both spellings map to the same yaml key so a project using either one works. What the value
becomes is the trainer's business -- a probability for yolov5, a boolean for classification --
which is why only the mapping is shared."""

BOOL_STRINGS = {'true': True, '1': True, 'yes': True, 'false': False, '0': False, 'no': False}
"""The spellings of a boolean the loop may send, in addition to a real ``bool``."""

_KINDS = ('int', 'float', 'bool', 'text', 'choice', 'id_value_map')


@dataclass(frozen=True)
class Parameter:
    """One knob a trainer accepts, as plain data.

    Build these with :func:`Int`, :func:`Float`, :func:`Bool`, :func:`Text`, :func:`Choice` or
    :func:`IdValueMap` rather than by hand -- they name the ``kind`` strings for you.
    """
    name: str
    kind: str
    default: Any = None
    required: bool = False
    minimum: float | None = None
    maximum: float | None = None
    exclusive_minimum: bool = False
    choices: tuple | None = None
    value_kind: str | None = None
    description: str = ''

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f'{self.name}: kind must be one of {_KINDS}, got {self.kind!r}')
        if self.kind == 'choice' and not self.choices:
            raise ValueError(f'{self.name}: a choice needs choices')

    def as_dict(self) -> dict[str, Any]:
        """The wire form, for reporting the schema to the loop."""
        return {key: value for key, value in asdict(self).items() if value not in (None, '', False)}


def Int(name: str, *, default: int | None = None, required: bool = False,
        minimum: int | None = None, maximum: int | None = None, description: str = '') -> Parameter:
    """A whole number. A fractional value is rejected rather than truncated."""
    return Parameter(name=name, kind='int', default=default, required=required,
                     minimum=minimum, maximum=maximum, description=description)


def Float(name: str, *, default: float | None = None, required: bool = False,
          minimum: float | None = None, maximum: float | None = None,
          exclusive_minimum: bool = False, description: str = '') -> Parameter:
    """A real number."""
    return Parameter(name=name, kind='float', default=default, required=required, minimum=minimum,
                     maximum=maximum, exclusive_minimum=exclusive_minimum, description=description)


def Bool(name: str, *, default: bool = False, description: str = '') -> Parameter:
    """A flag, accepting the strings in :data:`BOOL_STRINGS` as well as a real ``bool``."""
    return Parameter(name=name, kind='bool', default=default, description=description)


def Text(name: str, *, default: str = '', required: bool = False, description: str = '') -> Parameter:
    """A free-form string."""
    return Parameter(name=name, kind='text', default=default, required=required, description=description)


def Choice(name: str, *, choices: Sequence[Any], default: Any = None,
           value_kind: str = 'text', description: str = '') -> Parameter:
    """One of a fixed set of values."""
    return Parameter(name=name, kind='choice', default=default, choices=tuple(choices),
                     value_kind=value_kind, description=description)


def IdValueMap(name: str, *, value_kind: str = 'text', description: str = '') -> Parameter:
    """The loop's ``"id:value,id:value"`` encoding, as a tuple of pairs.

    Unset yields an empty tuple. The ids stay strings -- resolving them against a training's
    categories is the trainer's business.
    """
    return Parameter(name=name, kind='id_value_map', default=(), value_kind=value_kind,
                     description=description)


def validate(schema: Sequence[Parameter], hyperparameters: Mapping[str, Any], *,
             also_known: Collection[str] = ()) -> dict[str, Any]:
    """Coerce and range-check ``hyperparameters`` against ``schema``, before a training starts.

    Every declared name is present in the result: a value the loop did not send falls back to the
    declared default, and a declared ``required`` name that is missing is an error rather than a
    surprise later on.

    Names the schema does not declare are *reported, not rejected*. The loop's project-wide
    defaults reach every trainer whether or not it reads them, so failing here would refuse
    trainings that work today -- but the log line is what turns "my setting had no effect" from a
    mystery into one grep.

    :param also_known: Names handled elsewhere -- yaml-native knobs, for instance -- which should
        not be reported as unrecognised.
    :raises CriticalError: On a value that cannot be coerced, is out of range, or is required and
        absent.
    """
    values: dict[str, Any] = {}
    for param in schema:
        raw = hyperparameters.get(param.name)
        if raw is None or (isinstance(raw, str) and not raw.strip() and param.kind != 'text'):
            if param.required:
                raise CriticalError(f"hyperparameter '{param.name}' must be set")
            values[param.name] = param.default
            continue
        values[param.name] = _check_range(param, _coerce(param, raw))

    declared = {param.name for param in schema}
    unknown = sorted(set(hyperparameters) - declared - set(REPORTED_NAMES) - set(also_known))
    if unknown:
        logger.warning('these hyperparameters are configured but not read by this trainer: %s',
                       ', '.join(unknown))
    return values


def merge_into_yaml(yaml_path: str, values: Mapping[str, Any], *, ignore: Collection[str] = ()) -> None:
    """Overwrite a yaml template's keys in place from ``values``, keeping each key's own type.

    For trainers configured through a yaml file the template is the list of knobs the training
    framework itself understands, and its existing value states the type to coerce to. Only keys
    already in the template are written -- a framework does not grow a knob because someone typed
    one -- so anything left over is named in the log instead of vanishing.

    Requires ``ruamel.yaml``, which every trainer using a yaml template already depends on. It is
    imported here rather than at module scope so the library needs no yaml dependency of its own.

    :param ignore: Names the caller handles itself, which are not "left over".
    :raises CriticalError: On an unwritable file or a template value of unsupported type.
    """
    try:  # pylint: disable=import-outside-toplevel
        from ruamel.yaml import YAML  # noqa: PLC0415
        from ruamel.yaml.scalarbool import ScalarBoolean  # noqa: PLC0415
        from ruamel.yaml.scalarfloat import ScalarFloat  # noqa: PLC0415
        from ruamel.yaml.scalarint import ScalarInt  # noqa: PLC0415
        from ruamel.yaml.scalarstring import LiteralScalarString  # noqa: PLC0415
    except ImportError as error:
        raise CriticalError('merge_into_yaml needs ruamel.yaml, which is not installed') from error

    yaml = YAML()
    with open(yaml_path) as file:
        content = yaml.load(file)

    for name in content:
        value = values.get(name)
        if value is None:
            continue
        reference = content[name]
        if isinstance(reference, (LiteralScalarString, str)):
            content[name] = str(value)
        elif isinstance(reference, (ScalarBoolean, bool)):
            content[name] = bool(value)
        elif isinstance(reference, (ScalarFloat, float)):
            content[name] = float(value)
        elif isinstance(reference, (ScalarInt, int)):
            content[name] = int(value)
        else:
            raise CriticalError(f'{yaml_path}: cannot set {name}, unknown type {type(reference)}')

    left_over = sorted(set(values) - set(content) - set(REPORTED_NAMES) - set(ignore))
    if left_over:
        logger.warning('%s has no place for these hyperparameters, they are ignored: %s',
                       yaml_path, ', '.join(left_over))

    try:
        with open(yaml_path, 'w') as file:
            yaml.dump(content, file)
    except Exception as error:
        raise CriticalError(f'could not write {yaml_path}: {error}') from None

    logger.info('%s after update: %s', yaml_path, content)


def _coerce(param: Parameter, raw: Any) -> Any:
    kind = param.value_kind if param.kind == 'choice' else param.kind
    if param.kind == 'id_value_map':
        return _coerce_pairs(param, raw)

    if kind == 'bool':
        value: Any = _coerce_bool(param, raw)
    elif kind == 'text':
        value = str(raw)
    elif kind == 'int':
        value = _coerce_int(param, raw)
    elif kind == 'float':
        try:
            value = float(raw)
        except (TypeError, ValueError) as error:
            raise CriticalError(f"hyperparameter '{param.name}' must be a number, got {raw!r}") from error
    else:
        raise CriticalError(f"hyperparameter '{param.name}' has unsupported kind {kind!r}")

    if param.kind == 'choice' and value not in (param.choices or ()):
        raise CriticalError(f"hyperparameter '{param.name}' must be one of "
                            f'{list(param.choices or ())}, got {value!r}')
    return value


def _coerce_bool(param: Parameter, raw: Any) -> bool:
    if not isinstance(raw, str):
        return bool(raw)
    if (key := raw.strip().lower()) not in BOOL_STRINGS:
        raise CriticalError(f"hyperparameter '{param.name}' must be a boolean, got {raw!r}")
    return BOOL_STRINGS[key]


def _coerce_int(param: Parameter, raw: Any) -> int:
    try:
        number = float(raw)
    except (TypeError, ValueError) as error:
        raise CriticalError(f"hyperparameter '{param.name}' must be a whole number, "
                            f'got {raw!r}') from error
    if number != int(number):
        raise CriticalError(f"hyperparameter '{param.name}' must be a whole number, got {raw!r}")
    return int(number)


def _coerce_pairs(param: Parameter, raw: Any) -> tuple[tuple[str, Any], ...]:
    pairs: list[tuple[str, Any]] = []
    for entry in (part.strip() for part in str(raw).split(',')):
        if not entry:
            continue
        parts = entry.split(':')
        if len(parts) != 2:
            raise CriticalError(f"hyperparameter '{param.name}': expected \"id:value\", "
                                f'got {entry!r}')
        key, value = parts[0].strip(), parts[1].strip()
        if param.value_kind == 'float':
            value = _coerce(Parameter(name=f'{param.name}[{key}]', kind='float'), value)
        pairs.append((key, value))
    return tuple(pairs)


def _check_range(param: Parameter, value: Any) -> Any:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return value
    if param.minimum is not None:
        if param.exclusive_minimum and value <= param.minimum:
            raise CriticalError(f"hyperparameter '{param.name}' must be > {param.minimum}, got {value}")
        if not param.exclusive_minimum and value < param.minimum:
            raise CriticalError(f"hyperparameter '{param.name}' must be >= {param.minimum}, got {value}")
    if param.maximum is not None and value > param.maximum:
        raise CriticalError(f"hyperparameter '{param.name}' must be <= {param.maximum}, got {value}")
    return value
