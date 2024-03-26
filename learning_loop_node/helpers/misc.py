"""original copied from https://quantlane.com/blog/ensure-asyncio-task-exceptions-get-logged/"""
import asyncio
import functools
import json
import logging
import os
import shutil
import sys
from dataclasses import asdict
from glob import glob
from time import perf_counter
from typing import Any, Coroutine, List, Optional, Tuple, TypeVar
from uuid import UUID, uuid4

import pynvml

from ..data_classes import Context, SocketResponse, Training
from ..globals import GLOBALS

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


def get_free_memory_mb() -> float:  # NOTE used by yolov5
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    free = float(info.free) / 1024 / 1024
    return free


async def is_valid_image(filename: str, check_jpeg: bool) -> bool:
    if not os.path.isfile(filename) or os.path.getsize(filename) == 0:
        return False
    if not check_jpeg:
        return True

    info = await asyncio.create_subprocess_shell(f'jpeginfo -c {filename}',
                                                 stdout=asyncio.subprocess.PIPE,
                                                 stderr=asyncio.subprocess.PIPE)
    out, _ = await info.communicate()
    return "OK" in out.decode()


async def delete_corrupt_images(image_folder: str, check_jpeg: bool = False) -> None:
    logging.info('deleting corrupt images')
    n_deleted = 0
    for image in glob(f'{image_folder}/*.jpg'):
        if not await is_valid_image(image, check_jpeg):
            logging.debug(f'  deleting image {image}')
            os.remove(image)
            n_deleted += 1

    logging.info(f'deleted {n_deleted} images')


def create_resource_paths(organization_name: str, project_name: str, image_ids: List[str]) -> Tuple[List[str], List[str]]:
    return [f'/{organization_name}/projects/{project_name}/images/{id}/main' for id in image_ids], image_ids


def create_image_folder(project_folder: str) -> str:
    image_folder = f'{project_folder}/images'
    os.makedirs(image_folder, exist_ok=True)
    return image_folder


def read_or_create_uuid(identifier: str) -> str:
    identifier = identifier.lower().replace(' ', '_')
    uuids = {}
    os.makedirs(GLOBALS.data_folder, exist_ok=True)
    file_path = f'{GLOBALS.data_folder}/uuids.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            uuids = json.load(f)

    uuid = uuids.get(identifier, None)
    if not uuid:
        uuid = str(uuid4())
        uuids[identifier] = uuid
        with open(file_path, 'w') as f:
            json.dump(uuids, f)
    return uuid


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
            if isinstance(value, bool):
                return asdict(SocketResponse.from_bool(value))
            if isinstance(value, SocketResponse):
                return value
            if (args[0] in ['connect', 'disconnect', 'connect_error']):
                return value
            if value is None:
                return None

            raise Exception(
                f"Return type for sio must be str, bool, SocketResponse or None', but was {type(value)}'")
        except Exception as e:
            logging.exception(f'An error occured for {args[0]}')

            return asdict(SocketResponse.for_failure(str(e)))

    return wrapper_ensure_socket_response


def is_valid_uuid4(val):
    if not val:
        return False
    try:
        _ = UUID(str(val)).version
        return True
    except ValueError:
        return False


def create_project_folder(context: Context) -> str:
    project_folder = f'{GLOBALS.data_folder}/{context.organization}/{context.project}'
    os.makedirs(project_folder, exist_ok=True)
    return project_folder


def activate_asyncio_warnings() -> None:
    '''Produce warnings for coroutines which take too long on the main loop and hence clog the event loop'''
    try:
        if sys.version_info.major >= 3 and sys.version_info.minor >= 7:  # most
            loop = asyncio.get_running_loop()
        else:
            loop = asyncio.get_event_loop()

        loop.set_debug(True)
        loop.slow_callback_duration = 0.2
        logging.info('activated asyncio warnings')
    except Exception:
        logging.exception('could not activate asyncio warnings. Exception:')


def images_for_ids(image_ids, image_folder) -> List[str]:
    logging.info(f'### Going to get images for {len(image_ids)} images ids')
    start = perf_counter()
    images = [img for img in glob(f'{image_folder}/**/*.*', recursive=True)
              if os.path.splitext(os.path.basename(img))[0] in image_ids]
    end = perf_counter()
    logging.info(f'found {len(images)} images for {len(image_ids)} image ids, which took {end-start:0.2f} seconds')
    return images


def generate_training(project_folder: str, context: Context) -> Training:
    training_uuid = str(uuid4())
    return Training(
        id=training_uuid,
        context=context,
        project_folder=project_folder,
        images_folder=create_image_folder(project_folder),
        training_folder=create_training_folder(project_folder, training_uuid)
    )


def delete_all_training_folders(project_folder: str):
    if not os.path.exists(f'{project_folder}/trainings'):
        return
    for uuid in os.listdir(f'{project_folder}/trainings'):
        shutil.rmtree(f'{project_folder}/trainings/{uuid}', ignore_errors=True)


def create_training_folder(project_folder: str, trainings_id: str) -> str:
    training_folder = f'{project_folder}/trainings/{trainings_id}'
    os.makedirs(training_folder, exist_ok=True)
    return training_folder
