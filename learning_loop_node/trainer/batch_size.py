"""Choosing a batch size by probing, rather than configuring one.

The trainer supplies a ``fits`` predicate that runs a representative step at doubling sizes.
Only powers of two are visited, so equal hardware yields an equal recipe. Nothing here imports a
deep-learning framework.

Adapted from PyTorch Lightning's ``BatchSizeFinder`` (power-scaling mode).
Copyright The Lightning AI team. Licensed under the Apache License, Version 2.0.
https://github.com/Lightning-AI/pytorch-lightning
"""

import logging
from collections.abc import Callable

from .exceptions import InsufficientMemoryError

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 1024
"""Where a search stops when its caller sets no bound of its own."""

NO_GPU_BATCH_SIZE = 8
"""Batch size used when there is no GPU to probe."""

MIN_TRAIN_STEPS_PER_EPOCH = 8
"""Fewest optimizer steps an epoch must have; :func:`dataset_limit` is derived from it."""


def find_batch_size(fits: Callable[[int], bool], *, limit: int) -> int:
    """Return the largest power-of-two batch size that fits, never exceeding ``limit``.

    :param fits: Runs a representative probe; ``False`` on out-of-memory.
    :raises InsufficientMemoryError: If not even a batch size of 1 fits.
    """
    limit = smaller_pot(limit)
    if not fits(1):
        raise InsufficientMemoryError('batch size 1 does not fit in memory')

    size = 1
    while size < limit and fits(size * 2):
        size *= 2

    return size


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
