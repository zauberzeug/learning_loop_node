"""Which images a training selects its models on.

A node selects models (best checkpoint, early stopping, confidence threshold) on
`Training.val_images()` and reads `Training.test_images()` only for the counts it reports. It never
decides between the two itself: `val_images()` always holds a set to select on, which is the test
images unless the training has `val` images of its own. A node may notice that both hold the same
images and spare itself a second pass, but nothing more.

Where the `val` images come from, in this order:

1. The loop sent a dedicated validation set (`set == 'val'`); it is used as it is.
2. The `compute_validation_set` hyperparameter is true: `assign_validation_set` moves about a tenth of the
   train images to `val`, chosen by a hash of the image id, so an image stays on its side when new
   images arrive or a training is continued.
3. Otherwise there are no `val` images and `val_images()` falls back to the test images, so the
   model is selected on the test set.

A training without test images fails before it starts (`require_test_images`): it would have
nothing to select its models on, or nothing to report their counts on.
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


def compute_validation_set(hyperparameters: Mapping[str, Any]) -> bool:
    """Whether `compute_validation_set` is set, defaulting to false; anything but a boolean is a CriticalError."""
    value = hyperparameters.get(COMPUTE_VALIDATION_SET, False)
    if not isinstance(value, bool):
        raise CriticalError(f"'{COMPUTE_VALIDATION_SET}' must be true or false, got {value!r}")
    return value


def require_test_images(image_data: list[dict]) -> None:
    """Raise a CriticalError if none of the images is a test image."""
    if not any(image['set'] == 'test' for image in image_data):
        raise CriticalError('The training has no test images to select its models on or to report them on, '
                            'add test images')


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
