"""Choosing a batch size by probing, rather than configuring one.

The trainer supplies a ``fits`` predicate that runs a representative step at doubling sizes.
Only powers of two are visited, so equal hardware yields an equal recipe. Nothing here imports a
deep-learning framework.

Adapted from PyTorch Lightning's ``BatchSizeFinder`` (power-scaling mode).
Copyright The Lightning AI team. Licensed under the Apache License, Version 2.0.
https://github.com/Lightning-AI/pytorch-lightning
"""

import logging
from collections.abc import Callable, Mapping
from typing import Any

from .exceptions import InsufficientMemoryError

logger = logging.getLogger(__name__)

BATCH_SIZE = 'batch_size'
"""The hyperparameter every node reads its `measure_batch_size` argument out of.

Named here so the nodes agree on the spelling, and here rather than in :mod:`.cuda` so that a
hyperparameter parser can read it without pulling torch in. 0 or absent means the card decides.
"""

MAX_BATCH_SIZE = 1024
"""Where a search stops when its caller sets no bound of its own."""

NO_GPU_BATCH_SIZE = 8
"""Batch size used when there is no GPU to probe."""

MIN_TRAIN_STEPS_PER_EPOCH = 8
"""Fewest optimizer steps an epoch must have; :func:`dataset_limit` is derived from it."""


def find_batch_size(fits: Callable[[int], bool], *, limit: int, minimum: int = 1,
                    candidate: int = 0) -> int:
    """Return the largest batch size that fits, never exceeding ``limit``.

    Powers of two, so equal hardware and equal hyperparameters yield an equal recipe — plus
    ``candidate``, which is the one size outside that set this will return, and only when somebody
    named it and it then measured.

    :param fits: Runs a representative probe; ``False`` on out-of-memory.
    :param limit: Upper bound; the doubling stops at the largest power of two within it.
    :param minimum: Smallest size to try, rounded down to a power of two. Raise it above one for a
        step that cannot run on a single sample at all — BatchNorm over a 1x1 feature map, a
        validation pass that halves the batch — where a failure at one says nothing about memory.
    :param candidate: An exact size, tried once the doubling has reached its ceiling, so a size
        that was asked for is used as asked for rather than rounded down. Ignored unless it lies
        between that ceiling and ``limit``; a bound nobody named — one derived from the dataset,
        say — must not be passed here.
    :raises InsufficientMemoryError: If not even ``minimum`` fits.
    """
    minimum = smaller_pot(max(1, minimum))
    bound = max(limit, minimum)
    ceiling = max(smaller_pot(bound), minimum)

    if not fits(minimum):
        raise InsufficientMemoryError(f'batch size {minimum} does not fit in memory')

    size = minimum
    while size < ceiling and fits(size * 2):
        size *= 2

    if size == ceiling and ceiling < candidate <= bound and fits(candidate):
        size = candidate  # the doubling was not what stopped it, so the named size is reachable

    return size


def requested_batch_size(hyperparameters: Mapping[str, Any]) -> int:
    """The bound a training asked for, read out of the hyperparameters the loop sent.

    A field nobody filled in arrives as absent, ``None`` or ``''`` depending on where it came
    from, and all three mean the same thing: no bound of its own, the card decides alone. Read it
    through here rather than reaching into the dict, so every node agrees on that.

    :raises ValueError: If the value is there but is not a number.
    """
    return int(hyperparameters.get(BATCH_SIZE, 0) or 0)


def dataset_limit(sample_count: int) -> int:
    """The batch size ceiling that still leaves ``MIN_TRAIN_STEPS_PER_EPOCH`` steps per epoch."""
    if sample_count < 1:
        raise ValueError(f'sample_count must be >= 1, got {sample_count}')
    return max(1, sample_count // MIN_TRAIN_STEPS_PER_EPOCH)


def no_gpu_batch_size(limit: int, probe: str) -> int:
    """The batch size to fall back on when there is no GPU to probe."""
    batch_size = min(smaller_pot(limit), NO_GPU_BATCH_SIZE)
    logger.warning('%s: CUDA is unavailable; using batch size %d without probing', probe, batch_size)
    return batch_size


def smaller_pot(n: int) -> int:
    """The largest power of two that is <= ``n``."""
    if n < 1:
        raise ValueError(f'n must be >= 1, got {n}')
    return 1 << (n.bit_length() - 1)


def batch_count(sample_count: int, batch_size: int) -> int:
    """How many batches a loader yields, the last one possibly short."""
    return -(-sample_count // batch_size)


def is_out_of_memory(exception: BaseException) -> bool:
    """Whether the exception signals exhausted memory, on the GPU or the host.

    cuDNN and cuBLAS workspace failures raise a plain ``RuntimeError``, so the message has to be
    matched too.
    """
    if isinstance(exception, MemoryError):
        return True
    message = str(exception).lower()
    return any(text in message for text in ('out of memory', 'alloc_failed', 'cuda error: unknown error'))
