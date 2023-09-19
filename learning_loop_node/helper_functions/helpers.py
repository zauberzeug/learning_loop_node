"""original copied from https://quantlane.com/blog/ensure-asyncio-task-exceptions-get-logged/"""

import asyncio
import functools
import logging
from typing import Any, Coroutine, Optional, Tuple, TypeVar

import pynvml

T = TypeVar('T')


# This type annotation has to be quoted for Python < 3.9, see https://www.python.org/dev/peps/pep-0585/
def create_task(coroutine: Coroutine, *, loop: Optional[asyncio.AbstractEventLoop] = None, ) -> 'asyncio.Task':
    '''This helper function wraps a ``loop.create_task(coroutine())`` call and ensures there is
    an exception handler added to the resulting task. If the task raises an exception it is logged
    using the provided ``logger``, with additional context provided by ``message`` and optionally
    ``message_args``.'''

    logger = logging.getLogger(__name__)
    message = 'Task raised an exception'
    message_args = ()
    if loop is None:
        loop = asyncio.get_running_loop()
    task = loop.create_task(coroutine)
    task.add_done_callback(
        functools.partial(_handle_task_result, logger=logger, message=message, message_args=message_args)
    )
    return task


def _handle_task_result(
    task: asyncio.Task,
    *,
    logger: logging.Logger,
    message: str,
    message_args: Tuple[Any, ...] = (),
) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        pass  # Task cancellation should not be logged as an error.
    # Ad the pylint ignore: we want to handle all exceptions here so that the result of the task
    # is properly logged. There is no point re-raising the exception in this callback.
    except Exception:  # pylint: disable=broad-except
        logger.exception(message, *message_args)


def get_free_memory_mb():
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    free = float(info.free) / 1024 / 1024
    return free
