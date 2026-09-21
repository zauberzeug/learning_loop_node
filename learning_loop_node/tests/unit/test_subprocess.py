import asyncio
import multiprocessing
import os
import queue
import signal
import sys
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from ...trainer import subprocess as iterator_module
from ...trainer.exceptions import UnexpectedWorkerExitError
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


@pytest.mark.parametrize(('mode', 'exit_code'), [
    ('system_exit', 1),
    ('system_exit_zero', 0),
    ('native_exit', 7),
    ('kill', -signal.SIGKILL),
])
async def test_iterator_rejects_exit_without_completion(mode: str, exit_code: int) -> None:
    with pytest.raises(UnexpectedWorkerExitError, match=rf'exited with code {exit_code} .*IteratorDone'):
        await asyncio.wait_for(_collect_worker(_exit_without_completion, mode), timeout=10)


@pytest.mark.parametrize('error', [SystemExit(1), ValueError('worker failed')])
async def test_iterator_reports_failure_after_progress(error: BaseException) -> None:
    seen = []

    async def consume() -> None:
        async with iterator_cpu_bound(_fail_after_progress, error) as results:
            async for item in results:
                seen.append(item)

    expected_error = UnexpectedWorkerExitError if isinstance(error, SystemExit) else ValueError
    with pytest.raises(expected_error):
        await asyncio.wait_for(consume(), timeout=10)
    assert seen == [0]


async def test_iterator_accepts_completion_arriving_after_queue_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    context = multiprocessing.get_context('spawn')
    make_process = context.Process
    processes = []

    def record_process(*args: Any, **kwargs: Any) -> Any:
        process = make_process(*args, **kwargs)
        processes.append(process)
        return process

    to_thread = asyncio.to_thread
    first_poll = True

    async def delayed_timeout(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        nonlocal first_poll
        if first_poll:
            first_poll = False
            await to_thread(processes[0].join, 5)
            assert processes[0].exitcode == 0
            raise queue.Empty
        return await to_thread(func, *args, **kwargs)

    monkeypatch.setattr(context, 'Process', record_process)
    monkeypatch.setattr(iterator_module.asyncio, 'to_thread', delayed_timeout)

    assert await asyncio.wait_for(_collect_worker(counting, 0), timeout=10) == []


async def _collect_worker(it: Callable[..., Iterator[int]], *args: Any) -> list[int]:
    async with iterator_cpu_bound(it, *args) as results:
        return [item async for item in results]


def _exit_without_completion(mode: str) -> Iterator[int]:
    yield from ()
    if mode == 'system_exit':
        sys.exit(1)
    if mode == 'system_exit_zero':
        sys.exit(0)
    if mode == 'native_exit':
        os._exit(7)
    if mode == 'kill':
        os.kill(os.getpid(), signal.SIGKILL)
    raise AssertionError(f'Unexpected exit mode: {mode}')


def _fail_after_progress(error: BaseException) -> Iterator[int]:
    yield 0
    raise error
