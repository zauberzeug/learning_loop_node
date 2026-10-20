from collections.abc import Callable
from pathlib import Path

import pytest

from ...trainer import dataloader
from ...trainer.dataloader import (
    MAX_DATALOADER_WORKERS,
    MIN_DATALOADER_WORKERS,
    RESERVED_CPUS,
    available_cpus,
    default_dataloader_workers,
)

PretendCpus = Callable[..., None]


@pytest.mark.parametrize(('cpus', 'workers'), [(1, 2), (4, 2), (6, 4), (10, 8), (18, 16), (24, 16), (128, 16)])
def test_the_workers_leave_cpus_to_the_training_within_bounds(pretend_cpus: PretendCpus, cpus: int, workers: int):
    pretend_cpus(cpus)
    assert default_dataloader_workers() == workers


def test_the_bounds_hold_for_every_machine(pretend_cpus: PretendCpus):
    for cpus in range(1, 200):
        pretend_cpus(cpus)
        workers = default_dataloader_workers()
        assert MIN_DATALOADER_WORKERS <= workers <= MAX_DATALOADER_WORKERS
        assert workers == MIN_DATALOADER_WORKERS or workers <= cpus - RESERVED_CPUS


def test_the_affinity_decides_not_the_host(pretend_cpus: PretendCpus, monkeypatch: pytest.MonkeyPatch):
    pretend_cpus(6)
    monkeypatch.setattr(dataloader.os, 'cpu_count', lambda: 64)
    assert available_cpus() == 6


def test_a_cgroup_quota_caps_the_affinity(pretend_cpus: PretendCpus):
    pretend_cpus(24, quota='400000 100000')
    assert available_cpus() == 4


def test_a_fractional_quota_rounds_down_to_whole_cpus(pretend_cpus: PretendCpus):
    pretend_cpus(24, quota='250000 100000')
    assert available_cpus() == 2


def test_a_quota_below_one_cpu_still_leaves_one(pretend_cpus: PretendCpus):
    pretend_cpus(24, quota='50000 100000')
    assert available_cpus() == 1


def test_an_unlimited_quota_leaves_the_affinity(pretend_cpus: PretendCpus):
    pretend_cpus(24, quota='max 100000')
    assert available_cpus() == 24


def test_a_quota_larger_than_the_affinity_does_not_add_cpus(pretend_cpus: PretendCpus):
    pretend_cpus(4, quota='1600000 100000')
    assert available_cpus() == 4


def test_an_unreadable_quota_is_no_quota(pretend_cpus: PretendCpus):
    pretend_cpus(8, quota='garbage')
    assert available_cpus() == 8


def test_without_an_affinity_the_cpu_count_is_used(pretend_cpus: PretendCpus, monkeypatch: pytest.MonkeyPatch):
    pretend_cpus(8)
    monkeypatch.delattr(dataloader.os, 'sched_getaffinity', raising=False)
    monkeypatch.setattr(dataloader.os, 'cpu_count', lambda: 12)
    assert available_cpus() == 12


@pytest.fixture(name='pretend_cpus')
def pretend_cpus_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> PretendCpus:
    """Pretend this process may run on ``count`` CPUs, under the cgroup quota ``quota`` if given."""
    cpu_max = tmp_path / 'cpu.max'
    monkeypatch.setattr(dataloader, 'CGROUP_CPU_MAX', cpu_max)

    def pretend(count: int, *, quota: str | None = None) -> None:
        monkeypatch.setattr(dataloader.os, 'sched_getaffinity', lambda _: set(range(count)), raising=False)
        if quota is None:
            cpu_max.unlink(missing_ok=True)
        else:
            cpu_max.write_text(quota + '\n')

    return pretend
