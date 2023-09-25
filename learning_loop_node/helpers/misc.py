"""original copied from https://quantlane.com/blog/ensure-asyncio-task-exceptions-get-logged/"""
import asyncio
import functools
import logging
import os
from dataclasses import asdict
from typing import Any, Coroutine, List, Optional, Tuple, TypeVar

import pynvml

from ..data_classes import SocketResponse

T = TypeVar('T')


# This type annotation has to be quoted for Python < 3.9, see https://www.python.org/dev/peps/pep-0585/
def create_task(coroutine: Coroutine, *, loop: Optional[asyncio.AbstractEventLoop] = None, ) -> asyncio.Task:
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


def _handle_task_result(task: asyncio.Task,
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


def get_free_memory_mb() -> float:  # TODO check if this is used
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    free = float(info.free) / 1024 / 1024
    return free


def create_resource_paths(organization_name: str, project_name: str, image_ids: List[str]) -> Tuple[List[str], List[str]]:
    # TODO: experimental:
    return [f'/{organization_name}/projects/{project_name}/images/{id}/main' for id in image_ids], image_ids
    # if not image_ids:
    #     return [], []
    # url_ids: List[Tuple(str, str)] = [(f'/{organization_name}/projects/{project_name}/images/{id}/main', id)
    #                                   for id in image_ids]
    # urls, ids = list(map(list, zip(*url_ids)))

    # return urls, ids


def create_image_folder(project_folder: str) -> str:
    image_folder = f'{project_folder}/images'
    os.makedirs(image_folder, exist_ok=True)
    return image_folder


def ensure_socket_response(func):
    """Decorator to ensure that the return value of a socket.io event handler is a SocketResponse.

    Args:
        func (Callable): The socket.io event handler
    """
    @functools.wraps(func)
    async def wrapper_ensure_socket_response(*args, **kwargs):
        try:
            if asyncio.iscoroutinefunction(func):
                value = await func(*args, **kwargs)
            else:
                value = func(*args, **kwargs)

            if isinstance(value, str):
                return asdict(SocketResponse.for_success(value))
            elif isinstance(value, bool):
                return asdict(SocketResponse.from_bool(value))
            elif isinstance(value, SocketResponse):
                return value
            elif (args[0] in ['connect', 'disconnect', 'connect_error']):
                return value
            elif value is None:
                return None
            else:
                raise Exception(
                    f"Return type for sio must be str, bool, SocketResponse or None', but was {type(value)}'")
        except Exception as e:
            logging.exception(f'An error occured for {args[0]}')

            return asdict(SocketResponse.for_failure(str(e)))

    return wrapper_ensure_socket_response
