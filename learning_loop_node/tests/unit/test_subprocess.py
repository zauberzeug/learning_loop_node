import pytest

from ...trainer.subprocess import iterator_cpu_bound


def counting(limit: int):
    yield from range(limit)


def raising_after(limit: int):
    yield from range(limit)
    raise RuntimeError('the training crashed')


def empty():
    return iter(())


async def _collect(iterator) -> list:
    return [item async for item in iterator]


async def test_everything_the_generator_yields_arrives_in_order():
    async with iterator_cpu_bound(counting, 5) as iterator:
        assert await _collect(iterator) == [0, 1, 2, 3, 4]


async def test_a_generator_that_yields_nothing_simply_finishes():
    async with iterator_cpu_bound(empty) as iterator:
        assert await _collect(iterator) == []


async def test_a_failure_in_the_process_is_raised_in_the_caller():
    with pytest.raises(RuntimeError, match='the training crashed'):
        async with iterator_cpu_bound(raising_after, 2) as iterator:
            await _collect(iterator)


async def test_what_the_generator_produced_before_failing_still_arrives():
    received = []
    with pytest.raises(RuntimeError):
        async with iterator_cpu_bound(raising_after, 3) as iterator:
            async for item in iterator:
                received.append(item)
    assert received == [0, 1, 2]


async def test_leaving_early_does_not_leave_the_process_running():
    async with iterator_cpu_bound(counting, 1000) as iterator:
        async for item in iterator:
            if item == 2:
                break
    # reaching here without hanging is the test
