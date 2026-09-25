"""GPU memory budgeting and batch-size probing for trainers that train in-process.

``--vram-limit-gb`` -- gigabytes of the card this training may use -- becomes both the budget a
probe measures against and the cap that holds the process to it, on whichever GPU the calling
process is using. The cap is a share of the card's *total* memory, not of what is free.

This is the one module in the library that imports torch, which the package does not declare, so
only a trainer may import it. The search itself is in :mod:`~learning_loop_node.trainer.batch_size`.
"""
from __future__ import annotations

import gc
import logging
from argparse import ArgumentParser
from collections.abc import Callable

import torch

from .batch_size import (
    MAX_BATCH_SIZE,
    REQUESTED_BATCH_SIZE,
    VRAM_LIMIT_GB_FLAG,
    VRAM_LIMIT_GB_HELP,
    dataset_limit,
    find_batch_size,
    is_out_of_memory,
    no_gpu_batch_size,
    smaller_pot,
)

logger = logging.getLogger(__name__)

SAFETY_MARGIN = 0.05
"""Share of the budget held back while probing, against allocator fragmentation later on."""


def measure_batch_size(run_batch: Callable[[int], str | None], *, batch_size: int = 0,
                       sample_count: int | None = None, probe: str = 'batch-size probe',
                       minimum: int = 1, vram_limit_gb: float = 0,
                       on_out_of_memory: Callable[[], None] | None = None) -> int:
    """Settle a training's batch size against what it asked for and what the card allows.

    The whole of the decision, so that a trainer is left with only the step. Every training enters
    here, whatever shape its hyperparameters have; :func:`probe_batch_size` is for a probe with no
    requested size to honour, such as a detection pass bounded only by how many images there are.

    :param run_batch: Runs the batch; may return a detail to append to the log line.
    :param batch_size: What the training asked for, as carried in the
        :data:`~.batch_size.REQUESTED_BATCH_SIZE` hyperparameter: the largest batch it may use,
        measured rather than trusted. A size that fits is used as asked for, whether or not it is a
        power of two; one that does not becomes the largest power of two below it that does, rather
        than a training that runs out of memory partway through. 0 means the card decides alone.
    :param sample_count: Samples in the training split, when the caller knows it. The search is
        then bounded so an epoch keeps enough optimizer steps to mean something, and so a loader
        that drops its last partial batch cannot end up with no batch at all. This bound is never
        used as the exact candidate -- it is a heuristic, not a size anyone named.
    :param minimum: Smallest size to try; see :func:`probe_batch_size`.
    :param vram_limit_gb: The budget the safety margin is a share of; 0 means the whole card.
    :param on_out_of_memory: Runs after a trial ran out of memory, to drop what it left behind.
    :raises InsufficientMemoryError: If not even ``minimum`` fits.
    :raises ValueError: If the training asked for a negative batch size.
    """
    if batch_size < 0:
        raise ValueError(f'{REQUESTED_BATCH_SIZE} must be >= 0, got {batch_size}')

    limit = batch_size
    if sample_count is not None:
        limit = min(limit or MAX_BATCH_SIZE, dataset_limit(sample_count))

    return probe_batch_size(run_batch, probe=probe, limit=limit, candidate=batch_size,
                            minimum=minimum, vram_limit_gb=vram_limit_gb,
                            on_out_of_memory=on_out_of_memory)


def probe_batch_size(run_batch: Callable[[int], str | None], *, probe: str = 'batch-size probe',
                     limit: int = 0, candidate: int = 0, minimum: int = 1, vram_limit_gb: float = 0,
                     on_out_of_memory: Callable[[], None] | None = None) -> int:
    """Run :func:`~learning_loop_node.trainer.batch_size.find_batch_size` against a real card.

    This is the whole of a probe except the step itself: the margin, the search, telling an
    out-of-memory failure from a bug, and releasing what the trials left behind. A caller that
    builds a throwaway model supplies ``on_out_of_memory`` to drop what a failed trial left on the
    card.

    :param run_batch: Runs the batch; may return a detail to append to the log line.
    :param probe: Names this probe in the log, so a node running several stays readable.
    :param limit: Caps the search; 0 means
        :data:`~learning_loop_node.trainer.batch_size.MAX_BATCH_SIZE`.
    :param candidate: An exact size to try once the doubling has reached its ceiling; see
        ``find_batch_size``.
    :param minimum: Smallest size to try; see ``find_batch_size``.
    :param vram_limit_gb: The budget the safety margin is a share of; 0 means the whole card.
    :param on_out_of_memory: Runs after a trial ran out of memory, to drop what it left behind
        (an optimizer's gradients, say).
    :raises InsufficientMemoryError: If not even ``minimum`` fits.
    """
    bound = max(limit or MAX_BATCH_SIZE, minimum)

    if not torch.cuda.is_available():
        return max(smaller_pot(max(1, minimum)), no_gpu_batch_size(bound, probe))

    margin = reserve_margin(vram_limit_gb, probe=probe)
    try:
        fits = measured_fits(run_batch, probe=probe, on_out_of_memory=on_out_of_memory)
        chosen = find_batch_size(fits, limit=bound, minimum=minimum, candidate=candidate)
    finally:
        del margin
        free_cuda_memory()
    logger.info('%s: selected batch size %d (upper bound %d)', probe, chosen, bound)
    return chosen


def measured_fits(run_batch: Callable[[int], str | None], *, probe: str,
                  on_out_of_memory: Callable[[], None] | None = None) -> Callable[[int], bool]:
    """Wrap ``run_batch`` into the ``fits`` predicate ``find_batch_size`` searches with.

    An out-of-memory failure is the answer "does not fit"; anything else is re-raised. Both
    arrive as the same exception types, so they are told apart by
    :func:`~learning_loop_node.trainer.batch_size.is_out_of_memory`, not by ``except``.

    :param run_batch: Runs the batch; may return a detail to append to the log line.
    :param on_out_of_memory: Runs after a trial ran out of memory, to drop what it left behind
        (an optimizer's gradients, say).
    """
    def fits(batch_size: int) -> bool:
        free_cuda_memory()
        try:
            torch.cuda.reset_peak_memory_stats()
            detail = run_batch(batch_size)
            torch.cuda.synchronize()
            logger.info('%s: %d fits (peak %.2f GB, margin included)%s', probe, batch_size,
                        torch.cuda.max_memory_allocated() / 1024**3, f'; {detail}' if detail else '')
            return True
        except (torch.cuda.OutOfMemoryError, RuntimeError, MemoryError) as exc:
            if not isinstance(exc, torch.cuda.OutOfMemoryError) and not is_out_of_memory(exc):
                raise
            logger.info('%s: %d does not fit (%s)', probe, batch_size, type(exc).__name__)
            if on_out_of_memory is not None:
                on_out_of_memory()
            return False

    return fits


def reserve_margin(vram_limit_gb: float, *, probe: str) -> torch.Tensor:
    """Claim :data:`SAFETY_MARGIN` of the budget, so a trial competes against a smaller card.

    Keep the returned tensor alive for as long as the probe runs: releasing it hands the margin
    back, and the chosen size is no longer the size that was measured.

    :param vram_limit_gb: The budget the margin is a share of; 0 means the whole card.
    """
    margin_bytes = int(usable_memory_bytes(vram_limit_gb) * SAFETY_MARGIN)
    logger.info('%s: keeping %.0f MB free as a safety margin', probe, margin_bytes / 1024**2)
    return torch.empty(margin_bytes, dtype=torch.uint8, device='cuda')


def usable_memory_bytes(vram_limit_gb: float) -> int:
    """How much GPU memory this process may allocate, honouring :func:`limit_cuda_memory`.

    :param vram_limit_gb: 0 or less means the whole card.
    """
    total_bytes = torch.cuda.get_device_properties(None).total_memory
    if vram_limit_gb <= 0:
        return total_bytes
    return min(total_bytes, int(vram_limit_gb * 1024**3))


def add_vram_limit_argument(parser: ArgumentParser) -> None:
    """Give a spawned training script the same GPU budget flag its node has.

    The spawned process still has to call :func:`limit_cuda_memory` with it.
    """
    parser.add_argument(VRAM_LIMIT_GB_FLAG, type=float, default=0, help=VRAM_LIMIT_GB_HELP)


def limit_cuda_memory(vram_limit_gb: float) -> None:
    """Cap how much of the GPU this process may allocate, to ``vram_limit_gb`` gigabytes.

    Call this once per process that touches the GPU, a spawned training process included: the
    cap does not survive the spawn.

    :param vram_limit_gb: 0 or less means no cap.
    """
    if vram_limit_gb <= 0 or not torch.cuda.is_available():
        return

    total_bytes = torch.cuda.get_device_properties(None).total_memory
    fraction = vram_limit_gb * 1024**3 / total_bytes
    total_gb = total_bytes / 1024**3

    if fraction >= 1.0:
        logger.warning('VRAM limit of %.1f GB exceeds the card capacity of %.1f GB; not limiting',
                       vram_limit_gb, total_gb)
        return

    torch.cuda.set_per_process_memory_fraction(fraction, None)
    logger.info('Limiting VRAM usage to %.1f GB of %.1f GB (%.0f%%)', vram_limit_gb, total_gb, fraction * 100)


def free_cuda_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()
