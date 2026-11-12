"""Choosing a batch size by probing, rather than configuring one.

The trainer supplies a ``fits`` predicate that runs a representative step at doubling sizes; by
default the search then bisects between the last size that fit and the first that did not, so the
card is filled up to the batch rather than to the power of two below it. Nothing here imports a
deep-learning framework.

Adapted from PyTorch Lightning's ``BatchSizeFinder`` (its ``power`` and ``binsearch`` modes).
Copyright The Lightning AI team. Licensed under the Apache License, Version 2.0.
https://github.com/Lightning-AI/pytorch-lightning
"""

import logging
from collections.abc import Callable, Mapping
from enum import Enum
from typing import Any

from .exceptions import InsufficientMemoryError

logger = logging.getLogger(__name__)

REQUESTED_BATCH_SIZE = 'max_batch_size'
"""The hyperparameter every node reads its `measure_batch_size` argument out of.

0 or absent means the card decides. It is an input only: a trainer reports the size it settled on
under a different key, conventionally ``batch_size``, and never writes it back here.
"""

BATCH_SIZE_SEARCH = 'batch_size_search'
"""The hyperparameter naming the :class:`BatchSizeSearch` a training probes with; absent means binsearch."""


class BatchSizeSearch(str, Enum):
    """How :func:`find_batch_size` closes in on the largest batch that fits, named as Lightning names its modes."""

    BINSEARCH = 'binsearch'
    """Double until a size does not fit, then bisect below it; any size can result."""

    POWER = 'power'
    """Only double, so equal hardware yields an equal recipe: a power of two, ``minimum`` or ``candidate``."""


VRAM_LIMIT_GB_FLAG = '--vram-limit-gb'
"""The flag a trainer takes its GPU budget from; ``VRAM_LIMIT_GB`` follows from the name."""

VRAM_LIMIT_GB_HELP = ('Gigabytes of GPU memory a training may use. The batch size is probed against this limit '
                      'instead of the whole card, so a lower limit yields a smaller batch size rather than an '
                      'out-of-memory error. Use it to share a GPU or to keep headroom against fragmentation. '
                      "The limit is relative to the card's total memory, not to what is currently free. "
                      '0 (default) means no limit.')

MAX_BATCH_SIZE = 1024
"""Where a search stops when its caller sets no bound of its own."""

NO_GPU_BATCH_SIZE = 8
"""Batch size used when there is no GPU to probe."""

MIN_TRAIN_STEPS_PER_EPOCH = 8
"""Fewest optimizer steps an epoch must have; :func:`dataset_limit` is derived from it."""


def find_batch_size(fits: Callable[[int], bool], *, limit: int, minimum: int = 1,
                    candidate: int = 0, search: BatchSizeSearch = BatchSizeSearch.BINSEARCH) -> int:
    """Return the largest batch size that fits, never exceeding ``limit``.

    A :attr:`~BatchSizeSearch.POWER` search returns powers of two, plus ``minimum`` and
    ``candidate``, the two sizes outside that set it will return, and only because somebody named
    them and they then measured.

    :param fits: Runs a representative probe; ``False`` on out-of-memory.
    :param limit: Upper bound. A power search stops at the largest power of two within it; a
        binsearch doubles up to ``limit`` itself.
    :param minimum: Smallest size to try, as named; the doubling continues from the next power of
        two above it. Raise it above one for a step that cannot run on a single sample at all:
        BatchNorm over a 1x1 feature map, a validation pass that halves the batch.
    :param candidate: An exact size a power search tries once the doubling has reached its
        ceiling, so a size that was asked for is used as asked for rather than rounded down.
        Ignored unless it lies between that ceiling and ``limit``; a bound nobody named — one
        derived from the dataset, say — must not be passed here. A binsearch ignores it.
    :param search: Whether to bisect after the doubling or keep to powers of two.
    :raises InsufficientMemoryError: If not even ``minimum`` fits.
    :raises ValueError: If ``search`` names no :class:`BatchSizeSearch`.
    """
    if search not in (BatchSizeSearch.BINSEARCH, BatchSizeSearch.POWER):
        raise ValueError(f'unknown batch size search {search!r}')

    minimum = max(1, minimum)
    bound = max(limit, minimum)

    if not fits(minimum):
        raise InsufficientMemoryError(f'batch size {minimum} does not fit in memory')

    if search == BatchSizeSearch.BINSEARCH:
        return _binsearch(fits, minimum, bound)
    return _power_search(fits, minimum, bound, candidate)


def requested_batch_size(hyperparameters: Mapping[str, Any]) -> int:
    """The bound a training asked for, read out of the hyperparameters the loop sent.

    An unfilled field arrives as absent, ``None`` or ``''``, and all three mean no bound.

    :raises ValueError: If the value is there but is not a number.
    """
    return int(hyperparameters.get(REQUESTED_BATCH_SIZE, 0) or 0)


def requested_batch_size_search(hyperparameters: Mapping[str, Any]) -> BatchSizeSearch:
    """The :class:`BatchSizeSearch` a training asked for; absent, ``None`` or ``''`` mean binsearch.

    :raises ValueError: If the value names no search.
    """
    value = hyperparameters.get(BATCH_SIZE_SEARCH) or BatchSizeSearch.BINSEARCH
    try:
        return BatchSizeSearch(value)
    except ValueError as e:
        choices = ', '.join(search.value for search in BatchSizeSearch)
        raise ValueError(f'{BATCH_SIZE_SEARCH} must be one of {choices}, got {value!r}') from e


def dataset_limit(sample_count: int) -> int:
    """The batch size ceiling that still leaves ``MIN_TRAIN_STEPS_PER_EPOCH`` steps per epoch."""
    if sample_count < 1:
        raise ValueError(f'sample_count must be >= 1, got {sample_count}')
    return max(1, sample_count // MIN_TRAIN_STEPS_PER_EPOCH)


def no_gpu_batch_size(limit: int, probe: str, minimum: int = 1) -> int:
    """The batch size to fall back on when there is no GPU to probe; never below ``minimum``."""
    batch_size = max(min(smaller_pot(limit), NO_GPU_BATCH_SIZE), minimum)
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


def _binsearch(fits: Callable[[int], bool], low: int, bound: int) -> int:
    """Lightning's ``_run_binsearch_scaling`` from a size that fit, its ``max_val`` being ``bound``."""
    high = None
    while True:
        if high is None:
            if low >= bound:
                return low
            size = min(smaller_pot(low) * 2, bound)
        else:
            if high - low <= 1:
                return low
            size = (high + low) // 2
        if fits(size):
            low = size
        else:
            high = size


def _power_search(fits: Callable[[int], bool], minimum: int, bound: int, candidate: int) -> int:
    """Double from ``minimum``, a size that fit, up to the largest power of two within ``bound``."""
    ceiling = max(smaller_pot(bound), minimum)
    size = minimum
    while size < ceiling and fits(smaller_pot(size) * 2):
        size = smaller_pot(size) * 2

    if size == ceiling and ceiling < candidate <= bound and fits(candidate):
        size = candidate

    return size
