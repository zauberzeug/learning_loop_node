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
from collections.abc import Callable

import torch

from .batch_size import MAX_BATCH_SIZE, find_batch_size, is_out_of_memory, no_gpu_batch_size, smaller_pot

logger = logging.getLogger(__name__)

SAFETY_MARGIN = 0.05
"""Share of the budget held back while probing, against allocator fragmentation later on."""


def probe_batch_size(run_batch: Callable[[int], str | None], *, probe: str = 'batch-size probe',
                     limit: int = 0, vram_limit_gb: float = 0) -> int:
    """Find the largest power-of-two batch size ``run_batch`` fits into.

    For a probe whose measurement is one call. A probe that has to build a throwaway model first
    composes :func:`reserve_margin`, :func:`measured_fits` and ``find_batch_size`` itself.

    :param run_batch: Runs the batch; may return a detail to append to the log line.
    :param probe: Names this probe in the log, so a node running several stays readable.
    :param limit: Caps the search, rounded down to a power of two; 0 means
        :data:`~learning_loop_node.trainer.batch_size.MAX_BATCH_SIZE`.
    :param vram_limit_gb: The budget the safety margin is a share of; 0 means the whole card.
    :raises InsufficientMemoryError: If not even a batch size of 1 fits.
    """
    limit = smaller_pot(limit or MAX_BATCH_SIZE)

    if not torch.cuda.is_available():
        return no_gpu_batch_size(limit, probe)

    margin = reserve_margin(vram_limit_gb, probe=probe)
    try:
        chosen = find_batch_size(measured_fits(run_batch, probe=probe), limit=limit)
    finally:
        del margin
        free_cuda_memory()
    logger.info('%s: selected batch size %d (upper bound %d)', probe, chosen, limit)
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
