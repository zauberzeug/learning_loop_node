from collections.abc import Callable

import pytest

from ...trainer.batch_size import (
    MIN_TRAIN_STEPS_PER_EPOCH,
    batch_count,
    dataset_limit,
    find_batch_size,
    is_out_of_memory,
    no_gpu_batch_size,
    smaller_pot,
)


def _recording(capacity: int) -> tuple[Callable[[int], bool], list[int]]:
    """A `fits` predicate for a machine of `capacity`, plus the sizes it gets asked about."""
    calls: list[int] = []

    def fits(batch_size: int) -> bool:
        calls.append(batch_size)
        return batch_size <= capacity

    return fits, calls


def _fits_up_to(capacity: int) -> Callable[[int], bool]:
    fits, _ = _recording(capacity)
    return fits


def test_the_search_doubles_up_to_the_limit():
    fits, calls = _recording(1024)
    assert find_batch_size(fits, limit=16) == 16
    assert calls == [1, 2, 4, 8, 16]


def test_the_search_backs_off_to_the_last_size_that_fit():
    fits, calls = _recording(20)
    assert find_batch_size(fits, limit=64) == 16
    assert calls == [1, 2, 4, 8, 16, 32]  # probes 32, fails, keeps 16


def test_a_limit_that_is_not_a_power_of_two_is_rounded_down():
    assert find_batch_size(_fits_up_to(1024), limit=48) == 32
    assert find_batch_size(_fits_up_to(1024), limit=1) == 1


def test_a_machine_that_cannot_take_one_sample_is_an_error():
    fits, calls = _recording(0)
    with pytest.raises(RuntimeError, match='batch size 1 does not fit'):
        find_batch_size(fits, limit=64)
    assert calls == [1], 'must give up instead of probing larger sizes'


@pytest.mark.parametrize('capacity', range(1, 130))
def test_the_result_is_always_the_largest_power_of_two_that_fits(capacity: int):
    assert find_batch_size(_fits_up_to(capacity), limit=512) == smaller_pot(capacity)


def test_equal_hardware_yields_an_equal_recipe():
    """Only powers of two, so two machines of similar size train identically."""
    assert find_batch_size(_fits_up_to(37), limit=512) == find_batch_size(_fits_up_to(39), limit=512)


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


def test_the_dataset_limit_keeps_enough_steps_per_epoch():
    for sample_count in (8, 20, 47, 100, 1000, 118_000):
        limit = smaller_pot(dataset_limit(sample_count))
        assert sample_count // limit >= MIN_TRAIN_STEPS_PER_EPOCH


def test_the_dataset_limit_stays_usable_for_a_tiny_set():
    assert dataset_limit(7) == 1
    with pytest.raises(ValueError, match='sample_count must be >= 1'):
        dataset_limit(0)


def test_memory_still_decides_below_the_dataset_limit():
    """The bound is a ceiling only: a card that fits just 2 keeps training at 2."""
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
    """A trainer catching bare RuntimeError treats every crash as 'too big'; this does not."""
    assert not is_out_of_memory(RuntimeError('shape mismatch in forward pass'))
    assert not is_out_of_memory(ValueError('bad config'))


def test_the_host_running_out_of_memory_counts_too():
    assert is_out_of_memory(MemoryError())
