import asyncio
import os
import shutil
import time
import zipfile
from glob import glob
from typing import Callable

from ..helpers import environment_reader
from ..loop_communication import LoopCommunicator

PRODUCTION_LOOP_HOST = 'learning-loop.ai'


def assert_not_production_loop() -> None:
    """Refuse to run a destructive test fixture against the production Learning Loop.

    `LoopCommunicator` falls back to the production host when neither `LOOP_HOST` nor `HOST` is
    set, so a forgotten `.env` aims project creation and deletion at real customer data. Every
    fixture that creates or deletes a project should call this first.
    """
    host = environment_reader.host()
    if not host:
        raise AssertionError(
            'LOOP_HOST is not set. Destructive test fixtures refuse to run, because '
            f'LoopCommunicator would fall back to the production loop at {PRODUCTION_LOOP_HOST}. '
            'Point LOOP_HOST at a test instance, for example preview.learning-loop.ai.')
    if host == PRODUCTION_LOOP_HOST:
        raise AssertionError(
            f'LOOP_HOST is {host}, the production Learning Loop. Destructive test fixtures create '
            'and delete projects, so they refuse to run here. Point LOOP_HOST at a test instance, '
            'for example preview.learning-loop.ai.')


def get_files_in_folder(folder: str):
    files = [entry for entry in glob(f'{folder}/**/*', recursive=True) if os.path.isfile(entry)]
    files.sort()
    return files


async def get_latest_model_id(project: str) -> str:
    lc = LoopCommunicator()
    response = await lc.get(f'/zauberzeug/projects/{project}/trainings')
    await lc.shutdown()

    assert response.status_code == 200
    trainings = response.json()
    return trainings['charts'][0]['data'][0]['model_id']


def unzip(file_path, target_folder):
    shutil.rmtree(target_folder, ignore_errors=True)
    os.makedirs(target_folder)
    with zipfile.ZipFile(file_path, 'r') as zip_:
        zip_.extractall(target_folder)


async def condition(c_condition: Callable, *, timeout: float = 1.0, interval: float = 0.1):
    start = time.time()
    while not c_condition():
        if time.time() > start + timeout:
            raise TimeoutError(f'condition {c_condition} took longer than {timeout}s')
        await asyncio.sleep(interval)


def update_attributes(obj, **kwargs) -> None:
    if isinstance(obj, dict):
        _update_attribute_dict(obj, **kwargs)
    else:
        _update_attribute_class_instance(obj, **kwargs)


def _update_attribute_class_instance(obj, **kwargs) -> None:
    for key, value in kwargs.items():
        if hasattr(obj, key):
            setattr(obj, key, value)
        else:
            raise ValueError(f"Object of type '{type(obj)}' does not have a property '{key}'.")


def _update_attribute_dict(obj: dict, **kwargs) -> None:
    for key, value in kwargs.items():
        obj[key] = value
