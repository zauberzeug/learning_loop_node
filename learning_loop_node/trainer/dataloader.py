"""How many dataloader workers a training runs, derived from the CPUs this process may use.

Nothing here imports a deep-learning framework.
"""

import os
from pathlib import Path

DATALOADER_WORKERS_FLAG = '--dataloader-workers'
"""The flag a trainer takes its worker count from; ``DATALOADER_WORKERS`` follows from the name."""

DATALOADER_WORKERS_HELP = ('Dataloader worker processes a training uses. Defaults to the CPUs this process may run on, '
                           'minus two for the training process itself, between two and sixteen. Lower it on a host '
                           'whose CPUs are shared with other services, or when the host memory runs short.')

RESERVED_CPUS = 2
"""CPUs left to the training process itself, which collates and pins the batches and launches the kernels."""

MIN_DATALOADER_WORKERS = 2

MAX_DATALOADER_WORKERS = 16

CGROUP_CPU_MAX = Path('/sys/fs/cgroup/cpu.max')
"""Where cgroup v2 keeps a container's CPU quota, as set by ``docker run --cpus``."""


def default_dataloader_workers() -> int:
    """The worker count a training uses when ``DATALOADER_WORKERS`` names none."""
    return max(MIN_DATALOADER_WORKERS, min(available_cpus() - RESERVED_CPUS, MAX_DATALOADER_WORKERS))


def available_cpus() -> int:
    """The CPUs this process may run on: its affinity, capped by a cgroup CPU quota."""
    if hasattr(os, 'sched_getaffinity'):
        cpus = len(os.sched_getaffinity(0))
    else:
        cpus = os.cpu_count() or 1
    quota = _cgroup_cpu_quota()
    return cpus if quota is None else min(cpus, quota)


def _cgroup_cpu_quota() -> int | None:
    """Whole CPUs the cgroup quota allows, at least one; ``None`` without a quota."""
    try:
        quota, period = CGROUP_CPU_MAX.read_text().split()
    except (OSError, ValueError):
        return None
    if quota == 'max':
        return None
    return max(1, int(quota) // int(period))
