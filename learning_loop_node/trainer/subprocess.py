"""Run a blocking, CPU-bound generator in its own process without blocking the event loop.

The queue has ``maxsize=1``, so the producer never runs more than one item ahead of the consumer.
The context is spawn on every platform, so ``it`` and its arguments must be picklable and a
process that already initialised CUDA is never forked. An exception raised inside the process is
re-raised in the caller, and the process is killed if the caller leaves the context early.
"""
from __future__ import annotations

import asyncio
import logging
import multiprocessing
import queue
from collections.abc import AsyncGenerator, Callable, Iterator
from contextlib import asynccontextmanager
from multiprocessing.queues import Queue as MPQueue
from typing import Any, ParamSpec, TypeVar

from .exceptions import UnexpectedWorkerExitError

logger = logging.getLogger(__name__)

T = TypeVar('T')
P = ParamSpec('P')


@asynccontextmanager
async def iterator_cpu_bound(
    it: Callable[P, Iterator[T]],
    *args: P.args,
    **kwargs: P.kwargs,
) -> AsyncGenerator[AsyncGenerator[T, None], None]:
    iterator = _iterator_cpu_bound_inner(it, *args, **kwargs)
    try:
        yield iterator
    finally:
        await asyncio.shield(iterator.aclose())


async def _iterator_cpu_bound_inner(
    it: Callable[P, Iterator[T]],
    *args: P.args,
    **kwargs: P.kwargs,
) -> AsyncGenerator[T, None]:
    ctx = multiprocessing.get_context('spawn')
    state_queue: MPQueue[T | Exception | IteratorDone] = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_iterator_wrapper,
        args=(it, state_queue, args, kwargs),
        name='iterator_cpu_bound',
    )

    process.start()

    try:
        while True:
            try:
                item = await asyncio.to_thread(state_queue.get, True, 0.5)
            except queue.Empty:
                if process.is_alive():
                    continue
                # Completion may have arrived between the timeout and the exit check.
                try:
                    item = await asyncio.to_thread(state_queue.get_nowait)
                except queue.Empty as e:
                    raise UnexpectedWorkerExitError(
                        f'{process.name} exited with code {process.exitcode} without sending IteratorDone'
                    ) from e
            match item:
                case IteratorDone():
                    break
                case Exception() as e:
                    raise e
                case _ as other:
                    yield other
    finally:
        if process.is_alive():
            process.kill()
        process.join()


def _iterator_wrapper(
    it: Callable[..., Iterator[T]],
    state_queue: MPQueue[T | Exception | IteratorDone],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    try:
        for data in it(*args, **kwargs):
            state_queue.put(data)
    except Exception as e:
        logger.exception('iterator_cpu_bound child process failed')
        state_queue.put(e)

    state_queue.put(IteratorDone())


class IteratorDone:
    pass
