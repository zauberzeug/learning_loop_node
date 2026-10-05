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
)

logger = logging.getLogger(__name__)

SAFETY_MARGIN = 0.05
"""Share of the budget held back while probing, against allocator fragmentation later on."""

_budget_gb: float = 0
"""What :func:`limit_cuda_memory` capped this process to; 0 means the whole card."""


class ProbeStep:
    """A training step built for one batch-size probe and released once the probe is done.

    :func:`measure_batch_size` builds it through a factory, after the safety margin is reserved,
    and holds the only reference to it, so nothing built for the probe stays on the card for the
    training that follows. A subclass implements :meth:`run`; the other two are optional.
    """

    def run(self, batch_size: int) -> str | None:
        """Run one trial at ``batch_size``; may return a detail to append to the log line."""
        raise NotImplementedError

    def on_out_of_memory(self) -> None:
        """Drop what a trial that ran out of memory left behind (an optimizer's gradients, say)."""

    def release(self) -> None:
        """Undo what building the step changed outside it, such as moving the real model off the card."""


def measure_batch_size(build_step: Callable[[], ProbeStep], *, max_batch_size: int = 0,
                       sample_count: int | None = None, probe: str = 'batch-size probe',
                       minimum: int = 1) -> int:
    """Settle a training's batch size against what it asked for and what the card allows.

    Every training enters here, whatever shape its hyperparameters have; :func:`probe_batch_size` is
    for a probe with no requested size to honour, such as a detection pass bounded only by how many
    images there are. The step is built only once there is a card to probe.

    :param build_step: Builds the step the trials run; called once, after the margin is reserved.
    :param max_batch_size: What the training asked for, as carried in the
        :data:`~.batch_size.REQUESTED_BATCH_SIZE` hyperparameter: the largest batch it may use,
        measured rather than trusted. A size that fits is used as asked for, whether or not it is a
        power of two; one that does not becomes the largest power of two below it that does. 0 means
        the card decides alone. A request below ``minimum`` is raised to it, with a warning.
    :param sample_count: Samples in the training split, when the caller knows it. The search is
        then bounded by :func:`~.batch_size.dataset_limit`; that bound is never used as the exact
        candidate.
    :param probe: Names this probe in the log, so a node running several stays readable.
    :param minimum: Smallest size to try, taking precedence over ``max_batch_size``; see
        :func:`~.batch_size.find_batch_size`.
    :raises InsufficientMemoryError: If not even ``minimum`` fits.
    :raises ValueError: If the training asked for a negative batch size.
    """
    if max_batch_size < 0:
        raise ValueError(f'{REQUESTED_BATCH_SIZE} must be >= 0, got {max_batch_size}')
    if 0 < max_batch_size < minimum:
        logger.warning('%s: requested %s=%d is below the trainer minimum of %d; using %d', probe,
                       REQUESTED_BATCH_SIZE, max_batch_size, minimum, minimum)

    limit = max_batch_size
    if sample_count is not None:
        limit = min(limit or MAX_BATCH_SIZE, dataset_limit(sample_count))
        logger.info('%s: %d training samples allow at most %d per batch', probe, sample_count,
                    dataset_limit(sample_count))

    return _probe(build_step, probe=probe, limit=limit, candidate=max_batch_size, minimum=minimum)


def probe_batch_size(run_batch: Callable[[int], str | None], *, probe: str = 'batch-size probe',
                     limit: int = 0) -> int:
    """Find the largest batch that fits for a pass with no requested size, such as a detection pass.

    :param run_batch: Runs the batch; may return a detail to append to the log line.
    :param probe: Names this probe in the log, so a node running several stays readable.
    :param limit: Caps the search; 0 means
        :data:`~learning_loop_node.trainer.batch_size.MAX_BATCH_SIZE`.
    :raises InsufficientMemoryError: If not even a batch of one fits.
    """
    return _probe(lambda: _PassStep(run_batch), probe=probe, limit=limit)


class _PassStep(ProbeStep):
    """A plain callable as a step; there is nothing of its own to build or release."""

    def __init__(self, run_batch: Callable[[int], str | None]) -> None:
        self._run_batch = run_batch

    def run(self, batch_size: int) -> str | None:
        return self._run_batch(batch_size)


def _probe(build_step: Callable[[], ProbeStep], *, probe: str, limit: int, candidate: int = 0,
           minimum: int = 1) -> int:
    """Everything of a probe but the step: the GPU check, the margin, the search, the release.

    The margin is a share of the budget :func:`limit_cuda_memory` set in this process.
    """
    bound = max(limit or MAX_BATCH_SIZE, minimum)

    if not torch.cuda.is_available():
        return no_gpu_batch_size(bound, probe, minimum)

    margin = reserve_margin(probe=probe)
    try:
        chosen = _search(build_step, probe=probe, bound=bound, minimum=minimum, candidate=candidate)
    finally:
        del margin
        free_cuda_memory()
    logger.info('%s: selected batch size %d (upper bound %d)', probe, chosen, bound)
    return chosen


def _search(build_step: Callable[[], ProbeStep], *, probe: str, bound: int, minimum: int,
            candidate: int) -> int:
    """Build the step, search with it and release it; once this returns, nothing references it."""
    step = build_step()
    try:
        fits = measured_fits(step.run, probe=probe, on_out_of_memory=step.on_out_of_memory)
        return find_batch_size(fits, limit=bound, minimum=minimum, candidate=candidate)
    finally:
        step.release()


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


def reserve_margin(*, probe: str) -> torch.Tensor:
    """Claim :data:`SAFETY_MARGIN` of the budget, so a trial competes against a smaller card.

    Keep the returned tensor alive for as long as the probe runs: releasing it hands the margin
    back, and the chosen size is no longer the size that was measured.
    """
    margin_bytes = int(usable_memory_bytes() * SAFETY_MARGIN)
    logger.info('%s: keeping %.0f MB free as a safety margin', probe, margin_bytes / 1024**2)
    return torch.empty(margin_bytes, dtype=torch.uint8, device='cuda')


def usable_memory_bytes() -> int:
    """How much GPU memory this process may allocate, honouring :func:`limit_cuda_memory`."""
    if _budget_gb > 0:
        return int(_budget_gb * 1024**3)
    return torch.cuda.get_device_properties(None).total_memory


def add_vram_limit_argument(parser: ArgumentParser) -> None:
    """Give a spawned training script the same GPU budget flag its node has.

    The spawned process still has to call :func:`limit_cuda_memory` with it.
    """
    parser.add_argument(VRAM_LIMIT_GB_FLAG, type=float, default=0, help=VRAM_LIMIT_GB_HELP)


def limit_cuda_memory(vram_limit_gb: float) -> None:
    """Cap how much of the GPU this process may allocate, to ``vram_limit_gb`` gigabytes.

    Call this once per process that touches the GPU, a spawned training process included: the
    cap does not survive the spawn. A probe in this process then measures against the same budget.

    :param vram_limit_gb: 0 or less means no cap.
    """
    global _budget_gb  # pylint: disable=global-statement
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
    _budget_gb = vram_limit_gb
    logger.info('Limiting VRAM usage to %.1f GB of %.1f GB (%.0f%%)', vram_limit_gb, total_gb, fraction * 100)


def free_cuda_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()
