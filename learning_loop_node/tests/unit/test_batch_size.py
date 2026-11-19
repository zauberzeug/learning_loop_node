from collections.abc import Callable

import pytest

from ...trainer.batch_size import (
    MIN_TRAIN_STEPS_PER_EPOCH,
    REQUESTED_BATCH_SIZE,
    batch_count,
    dataset_limit,
    find_batch_size,
    is_out_of_memory,
    no_gpu_batch_size,
    requested_batch_size,
    smaller_pot,
)
from ...trainer.exceptions import InsufficientMemoryError


def test_a_binsearch_bisects_between_the_last_size_that_fit_and_the_first_that_did_not():
    fits, calls = _recording(20)
    assert find_batch_size(fits, limit=64) == 20
    assert calls == [1, 2, 4, 8, 16, 32, 24, 20, 22, 21]


def test_a_binsearch_doubles_up_to_the_limit_itself():
    fits, calls = _recording(1024)
    assert find_batch_size(fits, limit=48) == 48
    assert calls == [1, 2, 4, 8, 16, 32, 48], 'the limit is tried as named, not rounded down'


def test_a_binsearch_bisects_below_a_limit_that_does_not_fit():
    fits, calls = _recording(40)
    assert find_batch_size(fits, limit=48) == 40
    assert calls == [1, 2, 4, 8, 16, 32, 48, 40, 44, 42, 41]


def test_a_binsearch_from_a_minimum_rejoins_the_powers_of_two():
    ran: list[int] = []
    assert find_batch_size(_fits_up_to(16, ran), limit=64, minimum=3) == 16
    assert ran[:5] == [3, 4, 8, 16, 32]


def test_a_binsearch_keeps_a_minimum_above_the_limit():
    ran: list[int] = []
    assert find_batch_size(_fits_up_to(64, ran), limit=2, minimum=8) == 8
    assert ran == [8]


@pytest.mark.parametrize('capacity', range(1, 130))
def test_a_binsearch_finds_exactly_what_fits(capacity: int):
    assert find_batch_size(_fits_up_to(capacity), limit=512) == capacity


@pytest.mark.parametrize('limit', range(1, 70))
def test_a_binsearch_never_exceeds_the_limit(limit: int):
    ran: list[int] = []
    assert find_batch_size(_fits_up_to(40, ran), limit=limit) == min(40, limit)
    assert max(ran) <= limit


def test_a_binsearch_never_tries_a_size_twice():
    ran: list[int] = []
    find_batch_size(_fits_up_to(100, ran), limit=1000)
    assert len(ran) == len(set(ran))


def test_the_search_doubles_up_to_the_limit():
    fits, calls = _recording(1024)
    assert find_batch_size(fits, limit=16) == 16
    assert calls == [1, 2, 4, 8, 16]


def test_a_minimum_keeps_the_search_off_the_sizes_below_it():
    ran: list[int] = []
    assert find_batch_size(_fits_up_to(16, ran), limit=64, minimum=2) == 16
    assert min(ran) == 2, 'the smallest trial is the minimum, not one'


def test_a_minimum_that_fits_is_kept_when_the_next_power_of_two_does_not():
    assert find_batch_size(_fits_up_to(6, []), limit=64, minimum=6) == 6


def test_a_minimum_that_does_not_fit_reports_its_own_size():
    with pytest.raises(InsufficientMemoryError, match='batch size 4'):
        find_batch_size(_fits_up_to(2, []), limit=32, minimum=4)


def test_a_machine_that_cannot_take_one_sample_is_an_error():
    fits, calls = _recording(0)
    with pytest.raises(RuntimeError, match='batch size 1 does not fit'):
        find_batch_size(fits, limit=64)
    assert calls == [1], 'must give up instead of probing larger sizes'


def test_smaller_pot():
    assert [smaller_pot(n) for n in (1, 2, 3, 4, 7, 8, 15, 1293)] == [1, 2, 2, 4, 4, 8, 8, 1024]
    with pytest.raises(ValueError, match='n must be >= 1'):
        smaller_pot(0)


def test_batch_count_covers_the_whole_set_without_overshooting_by_a_batch():
    for sample_count in (1, 7, 8, 900, 5000):
        for batch_size in (1, 2, 8, 64, 512):
            covered = batch_count(sample_count, batch_size) * batch_size
            assert covered >= sample_count
            assert covered - sample_count < batch_size


@pytest.mark.parametrize('empty', [{}, {REQUESTED_BATCH_SIZE: None}, {REQUESTED_BATCH_SIZE: ''},
                                   {REQUESTED_BATCH_SIZE: 0}])
def test_an_unfilled_batch_size_means_no_bound(empty: dict):
    assert requested_batch_size(empty) == 0


def test_the_reported_batch_size_is_never_read_as_a_bound():
    """A trainer reports its settled size as `batch_size`; a resumed training must not read it as its bound."""
    assert REQUESTED_BATCH_SIZE != 'batch_size'
    assert requested_batch_size({'batch_size': 16}) == 0
    assert requested_batch_size({'batch_size': 16, REQUESTED_BATCH_SIZE: 64}) == 64


@pytest.mark.parametrize(('value', 'expected'), [('64', 64), (64, 64), (48.0, 48)])
def test_a_batch_size_is_read_whatever_the_loop_spelled_it_as(value: object, expected: int):
    assert requested_batch_size({REQUESTED_BATCH_SIZE: value}) == expected


def test_a_batch_size_that_is_not_a_number_is_an_error():
    with pytest.raises(ValueError):
        requested_batch_size({REQUESTED_BATCH_SIZE: 'lots'})


def test_the_dataset_limit_keeps_enough_steps_per_epoch():
    for sample_count in (8, 20, 47, 100, 1000, 118_000):
        limit = smaller_pot(dataset_limit(sample_count))
        assert sample_count // limit >= MIN_TRAIN_STEPS_PER_EPOCH


def test_the_dataset_limit_stays_usable_for_a_tiny_set():
    assert dataset_limit(7) == 1
    with pytest.raises(ValueError, match='sample_count must be >= 1'):
        dataset_limit(0)


def test_memory_still_decides_below_the_dataset_limit():
    assert find_batch_size(_fits_up_to(2), limit=dataset_limit(20)) == 2


def test_without_a_gpu_the_fallback_respects_the_limit():
    assert no_gpu_batch_size(1024, 'probe') == 8
    assert no_gpu_batch_size(2, 'probe') == 2


@pytest.mark.parametrize('message', [
    'CUDA out of memory. Tried to allocate 20.00 MiB',
    'cuDNN error: CUDNN_STATUS_ALLOC_FAILED',
    'CUDA error: unknown error',
])
def test_allocation_failures_are_recognised_however_they_surface(message: str):
    assert is_out_of_memory(RuntimeError(message))


def test_a_real_bug_is_not_mistaken_for_a_full_card():
    assert not is_out_of_memory(RuntimeError('shape mismatch in forward pass'))
    assert not is_out_of_memory(ValueError('bad config'))


def test_the_host_running_out_of_memory_counts_too():
    assert is_out_of_memory(MemoryError())


def _recording(capacity: int) -> tuple[Callable[[int], bool], list[int]]:
    """A `fits` predicate for a machine of `capacity`, plus the sizes it gets asked about."""
    calls: list[int] = []

    def fits(batch_size: int) -> bool:
        calls.append(batch_size)
        return batch_size <= capacity

    return fits, calls


def _fits_up_to(capacity: int, ran: list[int] | None = None) -> Callable[[int], bool]:
    """A `fits` predicate for a machine of `capacity`, appending every size asked about to `ran`."""
    fits, calls = _recording(capacity)
    if ran is None:
        return fits

    def recording_fits(batch_size: int) -> bool:
        result = fits(batch_size)
        ran[:] = calls
        return result

    return recording_fits
