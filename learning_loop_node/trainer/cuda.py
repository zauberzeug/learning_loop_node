"""GPU memory budgeting for trainers that train in-process.

A trainer that probes for its batch size needs two things a deployment can set: the budget the
probe measures against, and a cap that holds the process to it. Both come from one number --
gigabytes of the card this training may use -- which is what a node exposes as
``--vram-limit-gb``. It is how one GPU gets shared between processes, and how a training keeps
headroom against allocator fragmentation.

Unlike the rest of the library this module imports torch, and the package deliberately does not
declare it. Capping an allocator is a torch operation with no NVML equivalent, and declaring the
dependency would put an ML runtime into a library that is also installed on machines that train
nothing. Only a trainer imports this module, and a trainer brings torch already.

Note the cap is relative to the card's *total* memory, not to what is free, so it does not
protect against another process claiming memory first.
"""
from __future__ import annotations

import gc
import logging

import torch

logger = logging.getLogger(__name__)


def usable_memory_bytes(vram_limit_gb: float, device: int = 0) -> int:
    """How much GPU memory this process may allocate, honouring :func:`limit_cuda_memory`.

    :param vram_limit_gb: 0 or less means the whole card.
    """
    total_bytes = torch.cuda.get_device_properties(device).total_memory
    if vram_limit_gb <= 0:
        return total_bytes
    return min(total_bytes, int(vram_limit_gb * 1024**3))


def limit_cuda_memory(vram_limit_gb: float, device: int = 0) -> None:
    """Cap how much of the GPU this process may allocate, to ``vram_limit_gb`` gigabytes.

    Call this once per process that touches the GPU -- a spawned training process included,
    since the cap does not survive the spawn.

    :param vram_limit_gb: 0 or less means no cap.
    """
    if vram_limit_gb <= 0 or not torch.cuda.is_available():
        return

    total_bytes = torch.cuda.get_device_properties(device).total_memory
    fraction = vram_limit_gb * 1024**3 / total_bytes
    total_gb = total_bytes / 1024**3

    if fraction >= 1.0:
        logger.warning('VRAM limit of %.1f GB exceeds the card capacity of %.1f GB; not limiting',
                       vram_limit_gb, total_gb)
        return

    torch.cuda.set_per_process_memory_fraction(fraction, device)
    logger.info('Limiting VRAM usage to %.1f GB of %.1f GB (%.0f%%)', vram_limit_gb, total_gb, fraction * 100)


def free_cuda_memory() -> None:
    """Release cached CUDA memory, so a following allocation sees the real free space."""
    gc.collect()
    torch.cuda.empty_cache()
