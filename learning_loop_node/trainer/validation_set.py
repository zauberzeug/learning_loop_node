"""Which images a training selects its models on.

A node selects models (best checkpoint, early stopping, confidence threshold) on the `val` images
of `Training.image_data` if there are any, and reads `test` only for the counts it reports. Without
`val` images it selects on `test`, as every node did before validation sets existed.

Where the `val` images come from, in this order:

1. The loop sent a dedicated validation set (`set == 'val'`); it is used as it is.
2. The `compute_validation_set` hyperparameter is true: `assign_validation_set` moves about a tenth of the
   train images to `val`, chosen by a hash of the image id, so an image stays on its side when new
   images arrive or a training is continued.
3. Otherwise there are no `val` images.
"""

import hashlib
import logging
from collections.abc import Mapping
from typing import Any

from .exceptions import CriticalError

logger = logging.getLogger(__name__)

COMPUTE_VALIDATION_SET = 'compute_validation_set'
"""The hyperparameter that asks to split a validation set off the train images when the loop sent none."""

VALIDATION_FRACTION = 0.1

BOOL_STRINGS = {'true': True, '1': True, 'yes': True, 'false': False, '0': False, 'no': False}


def compute_validation_set(hyperparameters: Mapping[str, Any]) -> bool:
    """Whether `compute_validation_set` is set, defaulting to false; anything but a boolean is a CriticalError."""
    value = hyperparameters.get(COMPUTE_VALIDATION_SET)
    if value is None or value == '':
        return False
    if not isinstance(value, str):
        return bool(value)
    if (key := value.strip().lower()) not in BOOL_STRINGS:
        raise CriticalError(f"'{COMPUTE_VALIDATION_SET}' must be true or false, got {value!r}")
    return BOOL_STRINGS[key]


def assign_validation_set(image_data: list[dict], hyperparameters: Mapping[str, Any]) -> None:
    """Move part of the train images to `val` if `compute_validation_set` asks for it and the loop sent no `val` images.
    """
    if any(image['set'] == 'val' for image in image_data):
        logger.info('Selecting models on the validation set of the loop')
        return
    if not compute_validation_set(hyperparameters):
        return

    train_images = [image for image in image_data if image['set'] == 'train']
    if len(train_images) < 2:
        raise CriticalError(f'{len(train_images)} training image(s) are too few to split off a validation set, '
                            'add more training images')
    val_images = [image for image in train_images if _position(image) < VALIDATION_FRACTION] \
        or [min(train_images, key=_position)]
    for image in val_images:
        image['set'] = 'val'
    logger.info('Split %d of %d training images off as validation set', len(val_images), len(train_images))


def _position(image: dict) -> float:
    return int(hashlib.sha256(image['id'].encode()).hexdigest()[:8], 16) / 16**8
