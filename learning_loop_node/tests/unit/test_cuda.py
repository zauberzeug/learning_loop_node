"""Tests for the one library module that imports torch.

There is no torch in the dev extra and no GPU in CI, so a stand-in is installed under the name
``torch`` before the module is imported. Whether the cap actually holds, and what a real step
costs, can only be observed on a card.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
import types
import weakref
from collections.abc import Callable
from typing import Any

import pytest

from ...trainer.batch_size import MAX_BATCH_SIZE, NO_GPU_BATCH_SIZE
from ...trainer.exceptions import InsufficientMemoryError

MODULE = 'learning_loop_node.trainer.cuda'
GIB = 1024**3


# --- the budget and the cap ---

def test_no_limit_means_the_whole_card(load):
    cuda, _ = load(total_gb=8.0)
    assert cuda.usable_memory_bytes() == 8 * GIB
    cuda.limit_cuda_memory(0)
    assert cuda.usable_memory_bytes() == 8 * GIB


def test_a_limit_below_the_card_is_the_budget(load):
    cuda, _ = load(total_gb=8.0)
    cuda.limit_cuda_memory(6)
    assert cuda.usable_memory_bytes() == 6 * GIB


def test_a_limit_above_the_card_leaves_the_whole_card(load):
    cuda, _ = load(total_gb=8.0)
    cuda.limit_cuda_memory(16)
    assert cuda.usable_memory_bytes() == 8 * GIB


def test_the_cap_is_the_limits_share_of_the_card(load):
    cuda, fake = load(total_gb=8.0)
    cuda.limit_cuda_memory(2)
    assert fake.capped == [0.25]


def test_no_limit_caps_nothing(load):
    cuda, fake = load(total_gb=8.0)
    cuda.limit_cuda_memory(0)
    cuda.limit_cuda_memory(-1)
    assert fake.capped == []


def test_nothing_is_capped_without_cuda(load):
    cuda, fake = load(cuda_available=False)
    cuda.limit_cuda_memory(2)
    assert fake.capped == []


def test_a_spawned_script_takes_the_same_budget_flag_as_its_node(load):
    cuda, _ = load()
    parser = argparse.ArgumentParser()
    cuda.add_vram_limit_argument(parser)
    assert parser.parse_args([]).vram_limit_gb == 0
    assert parser.parse_args(['--vram-limit-gb', '6.5']).vram_limit_gb == 6.5


def test_a_limit_the_card_cannot_reach_warns_instead_of_capping(load, caplog):
    cuda, fake = load(total_gb=8.0)
    with caplog.at_level(logging.WARNING):
        cuda.limit_cuda_memory(8)
    assert fake.capped == []
    assert 'exceeds the card capacity' in caplog.text


def test_the_budget_and_the_cap_follow_the_current_device(load):
    cuda, fake = load(total_gb=8.0)
    cuda.usable_memory_bytes()
    cuda.limit_cuda_memory(2)
    assert fake.asked_devices and all(device is None for device in fake.asked_devices)


def test_freeing_empties_the_cache(load):
    cuda, fake = load()
    cuda.free_cuda_memory()
    assert fake.cache_clears == 1


# --- the safety margin ---

def test_the_margin_is_a_share_of_the_budget(load):
    cuda, fake = load(total_gb=8.0)
    cuda.limit_cuda_memory(4)
    cuda.reserve_margin(probe='probe')
    assert fake.allocated == [int(4 * GIB * cuda.SAFETY_MARGIN)]


def test_the_margin_is_a_share_of_the_whole_card_when_nothing_is_budgeted(load):
    cuda, fake = load(total_gb=8.0)
    cuda.reserve_margin(probe='probe')
    assert fake.allocated == [int(8 * GIB * cuda.SAFETY_MARGIN)]


# --- probing ---

def test_the_probe_keeps_the_largest_size_that_fits(load):
    cuda, fake = load()
    ran: list[int] = []
    assert cuda.probe_batch_size(_fits_up_to(16, fake, ran), limit=64) == 16
    assert ran == [1, 2, 4, 8, 16, 32], 'only powers of two, and one trial past the answer'


def test_the_probe_reserves_the_margin_before_it_measures(load):
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
    cuda, fake = load(cuda_available=False)
    ran: list[int] = []
    assert cuda.probe_batch_size(_fits_up_to(1024, fake, ran), limit=32) == NO_GPU_BATCH_SIZE
    assert cuda.probe_batch_size(_fits_up_to(1024, fake, ran), limit=2) == 2
    assert not ran, 'nothing may be run without a GPU'
    assert not fake.allocated, 'and no margin claimed on a card that is not there'


# --- settling a batch size from the hyperparameters ---

def test_a_requested_size_that_fits_is_used_as_named(load):
    cuda, fake = load()
    ran: list[int] = []
    assert cuda.measure_batch_size(_built(_fits_up_to(64, fake, ran)), max_batch_size=16) == 16
    assert ran[-1] == 16, 'and it was measured, not taken on trust'


def test_a_requested_size_that_does_not_fit_backs_off_instead_of_running_out_of_memory(load):
    cuda, fake = load()
    assert cuda.measure_batch_size(_built(_fits_up_to(8, fake, [])), max_batch_size=32) == 8


def test_a_requested_size_is_tried_even_when_it_is_not_a_power_of_two(load):
    cuda, fake = load()
    ran: list[int] = []
    assert cuda.measure_batch_size(_built(_fits_up_to(24, fake, ran)), max_batch_size=24) == 24
    assert _sizes(ran) == [1, 2, 4, 8, 16, 24], 'the named size only after the search reached its ceiling'


def test_a_named_size_that_does_not_fit_falls_back_to_the_power_of_two_below_it(load):
    cuda, fake = load()
    ran: list[int] = []
    assert cuda.measure_batch_size(_built(_fits_up_to(16, fake, ran)), max_batch_size=24) == 16
    assert ran[-1] == 24, 'it was tried, and it did not fit'


def test_a_named_size_is_not_tried_when_memory_stopped_the_search_earlier(load):
    cuda, fake = load()
    ran: list[int] = []
    assert cuda.measure_batch_size(_built(_fits_up_to(4, fake, ran)), max_batch_size=24) == 4
    assert 24 not in ran, 'if 16 does not fit, 24 cannot'


def test_an_absent_batch_size_leaves_the_bound_to_the_library(load):
    cuda, fake = load()
    assert cuda.measure_batch_size(_built(_fits_up_to(2048, fake, []))) == MAX_BATCH_SIZE


def test_a_batch_size_of_zero_means_measure(load):
    cuda, fake = load()
    ran: list[int] = []
    assert cuda.measure_batch_size(_built(_fits_up_to(16, fake, ran)), max_batch_size=0) == 16
    assert ran


def test_the_settled_size_is_only_returned(load):
    cuda, fake = load()
    settled = cuda.measure_batch_size(_built(_fits_up_to(16, fake, [])), max_batch_size=64)
    assert settled == 16, 'the caller reports it; nothing here stores it where a second call would read it'


def test_the_dataset_bounds_the_search_as_well(load):
    cuda, fake = load()
    # 80 samples leave room for 10 per step, rounded down to a power of two
    assert cuda.measure_batch_size(_built(_fits_up_to(1024, fake, [])), sample_count=80) == 8


def test_a_dataset_bound_is_never_used_as_a_candidate(load):
    cuda, fake = load()
    ran: list[int] = []
    cuda.measure_batch_size(_built(_fits_up_to(1024, fake, ran)), sample_count=80)
    assert 10 not in ran, 'samples // 8 is a heuristic, not a size anyone asked for'


def test_the_log_says_when_the_dataset_is_what_bounds_the_search(load, caplog):
    cuda, fake = load()
    with caplog.at_level(logging.INFO):
        cuda.measure_batch_size(_built(_fits_up_to(1024, fake, [])), sample_count=80)
    assert '80 training samples allow at most 10 per batch' in caplog.text


def test_the_tighter_of_the_request_and_the_dataset_wins(load):
    cuda, fake = load()
    assert cuda.measure_batch_size(_built(_fits_up_to(1024, fake, [])), max_batch_size=4,
                                   sample_count=8000) == 4
    assert cuda.measure_batch_size(_built(_fits_up_to(1024, fake, [])), max_batch_size=512,
                                   sample_count=80) == 8


def test_memory_still_decides_below_both_bounds(load):
    cuda, fake = load()
    assert cuda.measure_batch_size(_built(_fits_up_to(2, fake, [])), max_batch_size=64,
                                   sample_count=8000) == 2


def test_a_negative_batch_size_is_a_mistake_not_a_sentinel(load):
    cuda, fake = load()
    with pytest.raises(ValueError, match='max_batch_size'):
        cuda.measure_batch_size(_built(_fits_up_to(64, fake, [])), max_batch_size=-1)


def test_the_minimum_reaches_the_probe(load):
    cuda, fake = load()
    ran: list[int] = []
    cuda.measure_batch_size(_built(_fits_up_to(64, fake, ran)), minimum=4)
    assert ran[0] == 4


def test_a_request_below_the_minimum_is_raised_to_it_with_a_warning(load, caplog):
    cuda, fake = load()
    with caplog.at_level(logging.WARNING):
        assert cuda.measure_batch_size(_built(_fits_up_to(64, fake, [])), max_batch_size=1, minimum=2) == 2
    assert 'requested max_batch_size=1 is below the trainer minimum of 2; using 2' in caplog.text


def test_an_unset_request_does_not_warn_about_the_minimum(load, caplog):
    cuda, fake = load()
    with caplog.at_level(logging.WARNING):
        cuda.measure_batch_size(_built(_fits_up_to(64, fake, [])), minimum=2)
    assert 'below the trainer minimum' not in caplog.text


def test_a_minimum_above_the_request_still_gets_tried(load):
    cuda, fake = load()
    ran: list[int] = []
    assert cuda.measure_batch_size(_built(_fits_up_to(64, fake, ran)), max_batch_size=2, minimum=8) == 8
    assert _sizes(ran) == [8]


def test_without_a_gpu_the_fallback_respects_the_minimum(load):
    cuda, fake = load(cuda_available=False)
    ran: list[int] = []
    assert cuda.measure_batch_size(_built(_fits_up_to(1024, fake, ran)), max_batch_size=64, minimum=16) == 16
    assert cuda.measure_batch_size(_built(_fits_up_to(1024, fake, ran)), max_batch_size=64,
                                   minimum=3) == NO_GPU_BATCH_SIZE
    assert cuda.measure_batch_size(_built(_fits_up_to(1024, fake, ran)), max_batch_size=3, minimum=3) == 3
    assert not ran


# --- the step's lifecycle ---

def test_the_step_is_built_once_the_margin_is_reserved(load):
    cuda, fake = load()
    claimed: list[int] = []

    def build() -> _Step:
        claimed.extend(fake.allocated)
        return _Step(_fits_up_to(4, fake, []))

    cuda.measure_batch_size(build)
    assert claimed, 'so the step competes against the smaller card'


def test_no_step_is_built_without_a_gpu(load):
    cuda, _ = load(cuda_available=False)

    def build() -> _Step:
        raise AssertionError('nothing may be built for a card that is not there')

    assert cuda.measure_batch_size(build) == NO_GPU_BATCH_SIZE


def test_nothing_references_the_step_once_the_size_is_settled(load):
    cuda, fake = load()
    steps: list[weakref.ref] = []

    def build() -> _Step:
        step = _Step(_fits_up_to(4, fake, []))
        steps.append(weakref.ref(step))
        return step

    cuda.measure_batch_size(build)
    assert steps[0]() is None, 'the training that follows must not share the card with it'


def test_the_step_is_released_after_the_search(load):
    cuda, fake = load()
    step = _Step(_fits_up_to(4, fake, []))
    cuda.measure_batch_size(lambda: step)
    assert step.released == 1


def test_the_step_is_released_when_the_search_fails(load):
    cuda, fake = load()
    step = _Step(_fits_up_to(0, fake, []))
    with pytest.raises(InsufficientMemoryError):
        cuda.measure_batch_size(lambda: step)
    assert step.released == 1


def test_a_step_needs_to_implement_nothing_but_its_training_and_its_validation(load):
    cuda, fake = load()

    class Step(cuda.ProbeStep):
        def train_step(self, batch_size: int) -> None:
            _fits_up_to(4, fake, [])(batch_size)

        def val_step(self, batch_size: int) -> None:
            return None

    assert cuda.measure_batch_size(Step, max_batch_size=32) == 4


def test_a_step_without_a_validation_cannot_be_built(load):
    cuda, _ = load()

    class Step(cuda.ProbeStep):
        def train_step(self, batch_size: int) -> None:
            pass

    with pytest.raises(TypeError, match='val_step'):
        cuda.measure_batch_size(Step)


# --- what a trial runs ---

def test_a_trial_is_four_training_steps_and_then_a_validation(load):
    cuda, fake = load()
    step = _Step(_fits_up_to(1, fake, []))
    cuda.measure_batch_size(lambda: step, max_batch_size=2)
    assert step.calls[:5] == [('train', 1)] * 4 + [('val', 1)]
    assert cuda.DEFAULT_STEPS_PER_TRIAL == 4


def test_the_steps_per_trial_can_be_named(load):
    cuda, fake = load()
    step = _Step(_fits_up_to(1, fake, []))
    cuda.measure_batch_size(lambda: step, max_batch_size=2, steps_per_trial=2)
    assert step.calls[:3] == [('train', 1), ('train', 1), ('val', 1)]


def test_fewer_than_one_step_per_trial_is_a_mistake(load):
    cuda, fake = load()
    with pytest.raises(ValueError, match='steps_per_trial'):
        cuda.measure_batch_size(_built(_fits_up_to(64, fake, [])), steps_per_trial=0)


def test_a_size_that_runs_out_of_memory_in_a_later_step_does_not_fit(load):
    cuda, fake = load()
    taken: dict[int, int] = {}

    def run_batch(batch_size: int) -> None:
        taken[batch_size] = taken.get(batch_size, 0) + 1
        if batch_size > 4 and taken[batch_size] == 3:
            raise fake.OutOfMemoryError('the optimizer state arrived late')

    assert cuda.measure_batch_size(_built(run_batch), max_batch_size=32) == 4


def test_a_size_whose_validation_runs_out_of_memory_does_not_fit(load):
    cuda, fake = load()

    def validate(batch_size: int) -> None:
        if batch_size > 4:
            raise fake.OutOfMemoryError('validation is not autocast')

    step = _Step(_fits_up_to(1024, fake, []), validate)
    assert cuda.measure_batch_size(lambda: step, max_batch_size=32) == 4


def test_what_training_and_validation_report_is_logged_beside_the_peak(load, caplog):
    cuda, _ = load(peak_gb=3.0)

    class Step(cuda.ProbeStep):
        def train_step(self, batch_size: int) -> str:
            return '640 px'

        def val_step(self, batch_size: int) -> str:
            return 'validation at 1'

    with caplog.at_level(logging.INFO):
        cuda.measure_batch_size(Step, max_batch_size=1, probe='train probe')
    assert 'train probe: 1 fits (peak 3.00 GB, margin included); 640 px; validation at 1' in caplog.text


def test_a_detection_pass_runs_once_per_size(load):
    cuda, fake = load()
    ran: list[int] = []
    cuda.probe_batch_size(_fits_up_to(2, fake, ran), limit=4)
    assert ran == [1, 2, 4]


# --- cleaning up after a trial that did not fit ---

def test_the_out_of_memory_hook_runs_after_every_failed_trial(load):
    cuda, fake = load()
    ran: list[int] = []
    step = _Step(_fits_up_to(4, fake, ran))
    cuda.measure_batch_size(lambda: step, max_batch_size=32)
    assert step.dropped == [8], 'once, after the single trial that went over'


def test_the_out_of_memory_hook_does_not_run_for_a_bug(load):
    cuda, _ = load()

    def run_batch(_: int) -> None:
        raise RuntimeError('a real bug')

    step = _Step(run_batch)
    with pytest.raises(RuntimeError, match='a real bug'):
        cuda.measure_batch_size(lambda: step, max_batch_size=32)
    assert not step.dropped


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


# --- the card that is not there ---

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


def _built(run_batch: Callable[[int], None]) -> Callable[[], _Step]:
    """A factory for a step that trains with ``run_batch``."""
    return lambda: _Step(run_batch)


def _sizes(ran: list[int]) -> list[int]:
    """The sizes tried, in order, each once however many steps its trial took."""
    return list(dict.fromkeys(ran))


class _Step:
    """A step as ``measure_batch_size`` sees one, recording what was asked of it."""

    def __init__(self, run_batch: Callable[[int], None],
                 validate: Callable[[int], None] = lambda _: None) -> None:
        self._run_batch = run_batch
        self._validate = validate
        self.calls: list[tuple[str, int]] = []
        self.dropped: list[int] = []
        self.released = 0

    def train_step(self, batch_size: int) -> None:
        self.calls.append(('train', batch_size))
        self._run_batch(batch_size)

    def val_step(self, batch_size: int) -> None:
        self.calls.append(('val', batch_size))
        self._validate(batch_size)

    def on_out_of_memory(self) -> None:
        self.dropped.append(self.calls[-1][1])

    def release(self) -> None:
        self.released += 1


def _fits_up_to(largest: int, fake: _FakeTorch, ran: list[int]) -> Callable[[int], None]:
    """A step that runs out of memory above ``largest``, recording every size it was asked for."""
    def run_batch(batch_size: int) -> None:
        ran.append(batch_size)
        if batch_size > largest:
            raise fake.OutOfMemoryError('tried to allocate 20.00 GiB')
    return run_batch


class _FakeTorch:
    """What the module uses of torch, plus a record of what it asked for."""

    def __init__(self, *, cuda_available: bool, total_gb: float, peak_gb: float) -> None:
        self.capped: list[float] = []
        self.allocated: list[int] = []
        self.asked_devices: list[int | None] = []
        self.cache_clears = 0

        class OutOfMemoryError(RuntimeError):
            """Torch's own; note the module may not rely on its message."""

        self.OutOfMemoryError = OutOfMemoryError  # named as torch spells it
        self.module = types.ModuleType('torch')
        self.module.uint8 = 'uint8'  # type: ignore[attr-defined]
        self.module.empty = self._empty  # type: ignore[attr-defined]
        self.module.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
            OutOfMemoryError=OutOfMemoryError,
            is_available=lambda: cuda_available,
            empty_cache=self._empty_cache,
            get_device_properties=self._device_properties(total_gb),
            set_per_process_memory_fraction=self._cap,
            reset_peak_memory_stats=lambda: None,
            max_memory_allocated=lambda: int(peak_gb * GIB),
            synchronize=lambda: None,
        )

    def _empty(self, count: int, dtype: str, device: str) -> object:
        assert (dtype, device) == ('uint8', 'cuda')
        self.allocated.append(count)
        return object()

    def _empty_cache(self) -> None:
        self.cache_clears += 1

    def _device_properties(self, total_gb: float) -> Callable[[int | None], Any]:
        def get_device_properties(device: int | None = None):
            self.asked_devices.append(device)
            return types.SimpleNamespace(total_memory=int(total_gb * GIB))
        return get_device_properties

    def _cap(self, fraction: float, device: int | None = None) -> None:
        self.asked_devices.append(device)
        self.capped.append(fraction)
