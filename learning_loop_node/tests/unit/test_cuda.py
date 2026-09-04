"""Tests for the one library module that imports torch.

There is no torch in the dev extra, and no GPU in CI, so a stand-in is installed under the name
``torch`` before the module is imported. That covers the budgeting arithmetic and the guards --
everything this module decides. Whether the cap actually holds is torch's own business and can
only be observed on a card.
"""

import importlib
import logging
import sys
import types
from collections.abc import Callable
from typing import Any

import pytest

MODULE = 'learning_loop_node.trainer.cuda'
GIB = 1024**3


@pytest.fixture(name='load')
def load_fixture(monkeypatch: pytest.MonkeyPatch) -> Callable[..., tuple[Any, list[tuple[float, int]]]]:
    """Import the module against a fake torch; returns it and the fractions it capped with."""
    def load(*, cuda_available: bool = True, total_gb: float = 8.0):
        capped: list[tuple[float, int]] = []
        torch = types.ModuleType('torch')
        torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
            is_available=lambda: cuda_available,
            empty_cache=lambda: capped.append((-1.0, -1)),
            get_device_properties=lambda device: types.SimpleNamespace(total_memory=int(total_gb * GIB)),
            set_per_process_memory_fraction=lambda fraction, device: capped.append((fraction, device)),
        )
        monkeypatch.setitem(sys.modules, 'torch', torch)
        monkeypatch.delitem(sys.modules, MODULE, raising=False)
        return importlib.import_module(MODULE), capped

    yield load
    sys.modules.pop(MODULE, None)


def test_no_limit_means_the_whole_card(load):
    cuda, _ = load(total_gb=8.0)
    assert cuda.usable_memory_bytes(0) == 8 * GIB
    assert cuda.usable_memory_bytes(-1) == 8 * GIB


def test_a_limit_below_the_card_is_the_budget(load):
    cuda, _ = load(total_gb=8.0)
    assert cuda.usable_memory_bytes(6) == 6 * GIB


def test_a_limit_above_the_card_is_clamped_to_it(load):
    # otherwise the safety margin would be a share of memory that does not exist
    cuda, _ = load(total_gb=8.0)
    assert cuda.usable_memory_bytes(16) == 8 * GIB


def test_the_cap_is_the_limits_share_of_the_card(load):
    cuda, capped = load(total_gb=8.0)
    cuda.limit_cuda_memory(2)
    assert capped == [(0.25, 0)]


def test_no_limit_caps_nothing(load):
    cuda, capped = load(total_gb=8.0)
    cuda.limit_cuda_memory(0)
    cuda.limit_cuda_memory(-1)
    assert capped == []


def test_nothing_is_capped_without_cuda(load):
    cuda, capped = load(cuda_available=False)
    cuda.limit_cuda_memory(2)
    assert capped == []


def test_a_limit_the_card_cannot_reach_warns_instead_of_capping(load, caplog):
    # capping at a fraction >= 1 would be a no-op that reads as a limit having been applied
    cuda, capped = load(total_gb=8.0)
    with caplog.at_level(logging.WARNING):
        cuda.limit_cuda_memory(8)
    assert capped == []
    assert 'exceeds the card capacity' in caplog.text


def test_freeing_empties_the_cache(load):
    cuda, capped = load()
    cuda.free_cuda_memory()
    assert capped == [(-1.0, -1)]
