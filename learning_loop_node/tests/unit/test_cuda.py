"""Tests for the one library module that imports torch.

There is no torch in the dev extra, and no GPU in CI, so a stand-in is installed under the name
``torch`` before the module is imported. That covers the budgeting arithmetic, the guards and the
search around a probe -- everything this module decides. Whether the cap actually holds, and what
a real step costs, is torch's own business and can only be observed on a card.
"""

import importlib
import logging
import sys
import types
from collections.abc import Callable
from typing import Any

import pytest

from learning_loop_node.trainer.batch_size import MAX_BATCH_SIZE, NO_GPU_BATCH_SIZE
from learning_loop_node.trainer.exceptions import InsufficientMemoryError

MODULE = 'learning_loop_node.trainer.cuda'
GIB = 1024**3


class _FakeTorch:
    """What the module uses of torch, plus a record of what it asked for."""

    def __init__(self, *, cuda_available: bool, total_gb: float, peak_gb: float) -> None:
        self.capped: list[tuple[float, int]] = []
        self.allocated: list[int] = []

        class OutOfMemoryError(RuntimeError):
            """Torch's own; note the module may not rely on its message."""

        self.OutOfMemoryError = OutOfMemoryError  # named as torch spells it
        self.module = types.ModuleType('torch')
        self.module.uint8 = 'uint8'  # type: ignore[attr-defined]
        self.module.empty = self._empty  # type: ignore[attr-defined]
        self.module.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
            OutOfMemoryError=OutOfMemoryError,
            is_available=lambda: cuda_available,
            empty_cache=lambda: self.capped.append((-1.0, -1)),
            get_device_properties=lambda device: types.SimpleNamespace(total_memory=int(total_gb * GIB)),
            set_per_process_memory_fraction=lambda fraction, device: self.capped.append((fraction, device)),
            reset_peak_memory_stats=lambda: None,
            max_memory_allocated=lambda: int(peak_gb * GIB),
            synchronize=lambda: None,
        )

    def _empty(self, count: int, dtype: str, device: str) -> object:
        assert (dtype, device) == ('uint8', 'cuda')
        self.allocated.append(count)
        return object()


@pytest.fixture(name='load')
def load_fixture(monkeypatch: pytest.MonkeyPatch) -> Callable[..., tuple[Any, _FakeTorch]]:
    """Import the module against a fake torch; returns it and the stand-in it ran against."""
    def load(*, cuda_available: bool = True, total_gb: float = 8.0, peak_gb: float = 2.0):
        fake = _FakeTorch(cuda_available=cuda_available, total_gb=total_gb, peak_gb=peak_gb)
        monkeypatch.setitem(sys.modules, 'torch', fake.module)
        monkeypatch.delitem(sys.modules, MODULE, raising=False)
        return importlib.import_module(MODULE), fake

    yield load
    sys.modules.pop(MODULE, None)


def _fits_up_to(largest: int, fake: _FakeTorch, ran: list[int]) -> Callable[[int], None]:
    """A step that runs out of memory above ``largest``, recording every size it was asked for."""
    def run_batch(batch_size: int) -> None:
        ran.append(batch_size)
        if batch_size > largest:
            raise fake.OutOfMemoryError('tried to allocate 20.00 GiB')
    return run_batch


# --- the budget and the cap ---

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
    cuda, fake = load(total_gb=8.0)
    cuda.limit_cuda_memory(2)
    assert fake.capped == [(0.25, 0)]


def test_no_limit_caps_nothing(load):
    cuda, fake = load(total_gb=8.0)
    cuda.limit_cuda_memory(0)
    cuda.limit_cuda_memory(-1)
    assert fake.capped == []


def test_nothing_is_capped_without_cuda(load):
    cuda, fake = load(cuda_available=False)
    cuda.limit_cuda_memory(2)
    assert fake.capped == []


def test_a_limit_the_card_cannot_reach_warns_instead_of_capping(load, caplog):
    # capping at a fraction >= 1 would be a no-op that reads as a limit having been applied
    cuda, fake = load(total_gb=8.0)
    with caplog.at_level(logging.WARNING):
        cuda.limit_cuda_memory(8)
    assert fake.capped == []
    assert 'exceeds the card capacity' in caplog.text


def test_freeing_empties_the_cache(load):
    cuda, fake = load()
    cuda.free_cuda_memory()
    assert fake.capped == [(-1.0, -1)]


# --- the safety margin ---

def test_the_margin_is_a_share_of_the_budget(load):
    cuda, fake = load(total_gb=8.0)
    cuda.reserve_margin(4, probe='probe')
    assert fake.allocated == [int(4 * GIB * cuda.SAFETY_MARGIN)]


def test_the_margin_is_a_share_of_the_whole_card_when_nothing_is_budgeted(load):
    cuda, fake = load(total_gb=8.0)
    cuda.reserve_margin(0, probe='probe')
    assert fake.allocated == [int(8 * GIB * cuda.SAFETY_MARGIN)]


# --- probing ---

def test_the_probe_keeps_the_largest_size_that_fits(load):
    cuda, fake = load()
    ran: list[int] = []
    assert cuda.probe_batch_size(_fits_up_to(16, fake, ran), limit=64) == 16
    assert ran == [1, 2, 4, 8, 16, 32], 'only powers of two, and one trial past the answer'


def test_the_probe_reserves_the_margin_before_it_measures(load):
    # a trial that ran against the full card would choose a size the training cannot keep
    cuda, fake = load(total_gb=8.0)

    def run_batch(_: int) -> None:
        assert fake.allocated == [int(8 * GIB * cuda.SAFETY_MARGIN)]

    cuda.probe_batch_size(run_batch, limit=2)


def test_the_limit_is_rounded_down_to_a_power_of_two(load):
    cuda, fake = load()
    ran: list[int] = []
    assert cuda.probe_batch_size(_fits_up_to(1024, fake, ran), limit=12) == 8


def test_an_unset_limit_stops_at_the_maximum(load):
    cuda, fake = load()
    ran: list[int] = []
    assert cuda.probe_batch_size(_fits_up_to(2048, fake, ran)) == MAX_BATCH_SIZE


def test_a_probe_without_a_gpu_does_not_run_the_step(load):
    """With no card to measure, the fallback caps the batch and never runs the step."""
    cuda, fake = load(cuda_available=False)
    ran: list[int] = []
    assert cuda.probe_batch_size(_fits_up_to(1024, fake, ran), limit=32) == NO_GPU_BATCH_SIZE
    assert cuda.probe_batch_size(_fits_up_to(1024, fake, ran), limit=2) == 2
    assert not ran, 'nothing may be run without a GPU'
    assert not fake.allocated, 'and no margin claimed on a card that is not there'


def test_a_batch_size_of_one_that_does_not_fit_is_an_error(load):
    cuda, fake = load()
    ran: list[int] = []
    with pytest.raises(InsufficientMemoryError):
        cuda.probe_batch_size(_fits_up_to(0, fake, ran), limit=32)


def test_exhausted_host_memory_also_means_it_does_not_fit(load):
    cuda, _ = load()

    def run_batch(batch_size: int) -> None:
        if batch_size > 2:
            raise MemoryError

    assert cuda.probe_batch_size(run_batch, limit=32) == 2


def test_a_plain_runtime_error_saying_it_is_out_of_memory_does_not_fit(load):
    # cuDNN and cuBLAS workspaces arrive as bare RuntimeErrors
    cuda, _ = load()

    def run_batch(batch_size: int) -> None:
        if batch_size > 2:
            raise RuntimeError('cuDNN error: CUDNN_STATUS_ALLOC_FAILED')

    assert cuda.probe_batch_size(run_batch, limit=32) == 2


def test_torchs_own_error_needs_no_recognisable_message(load):
    cuda, fake = load()

    def run_batch(batch_size: int) -> None:
        if batch_size > 2:
            raise fake.OutOfMemoryError('see the memory summary')

    assert cuda.probe_batch_size(run_batch, limit=32) == 2


def test_a_failure_that_is_not_about_memory_is_a_bug_and_propagates(load):
    # a probe that swallows this would report the smallest batch size as the card's fault
    cuda, _ = load()

    def run_batch(batch_size: int) -> None:
        if batch_size > 2:
            raise RuntimeError('mat1 and mat2 shapes cannot be multiplied')

    with pytest.raises(RuntimeError, match='shapes cannot be multiplied'):
        cuda.probe_batch_size(run_batch, limit=32)


def test_what_the_step_reports_is_logged_beside_the_peak(load, caplog):
    cuda, _ = load(peak_gb=3.0)
    with caplog.at_level(logging.INFO):
        cuda.probe_batch_size(lambda _: 'validation peaked at 1.00 GB', probe='miniature epoch', limit=1)
    assert 'miniature epoch: 1 fits (peak 3.00 GB, margin included); validation peaked at 1.00 GB' in caplog.text


def test_only_a_trial_that_ran_out_of_memory_gets_cleaned_up_after(load):
    cuda, fake = load()
    cleaned: list[int] = []
    ran: list[int] = []
    fits = cuda.measured_fits(_fits_up_to(2, fake, ran), probe='probe',
                              on_out_of_memory=lambda: cleaned.append(len(ran)))

    assert [fits(size) for size in (1, 2, 4)] == [True, True, False]
    assert cleaned == [3], 'once, after the third trial'
