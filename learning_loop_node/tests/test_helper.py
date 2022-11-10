import asyncio
from glob import glob
import os
import time
from typing import Callable, List, Optional
import zipfile
from learning_loop_node.loop import loop
from urllib.parse import urljoin
from requests import Session
import aiohttp
import logging
from icecream import ic
import shutil


class LiveServerSession(Session):
    """https://stackoverflow.com/a/51026159/364388"""

    def __init__(self, *args, **kwargs):
        super(LiveServerSession, self).__init__(*args, **kwargs)
        self.prefix_url = loop.base_url

    def request(self, method, url, *args, **kwargs):
        url = urljoin(self.prefix_url, url)
        headers = {}
        if 'token' in url:
            return super(LiveServerSession, self).request(method, url, *args, **kwargs)

        headers = loop.get_headers()
        return super(LiveServerSession, self).request(method, url, headers=headers, *args, **kwargs)


def get_files_in_folder(folder: str):
    files = [entry for entry in glob(f'{folder}/**/*', recursive=True) if os.path.isfile(entry)]
    files.sort()
    return files


async def get_latest_model_id() -> str:
    async with loop.get(f'api/zauberzeug/projects/pytest/trainings') as response:
        assert response.status == 200
        trainings = await response.json()
        return trainings['charts'][0]['data'][0]['model_id']


async def assert_upload_model_with_id(file_paths: Optional[List[str]] = None, format: str = 'mocked', model_id: Optional[str] = None) -> str:
    data = prepare_formdata(file_paths)

    async with loop.put(f'api/zauberzeug/projects/pytest/models/{model_id}/{format}/file', data) as response:
        if response.status != 200:
            msg = f'unexpected status code {response.status} while putting model'
            logging.error(msg)
            raise (Exception(msg))
        model = await response.json()

        return model['id']


def prepare_formdata(file_paths: Optional[List[str]]) -> aiohttp.FormData:
    module_path = os.path.dirname(os.path.realpath(__file__))
    if not file_paths:
        file_paths = [f'{module_path}/test_data/file_1.txt',
                      f'{module_path}/test_data/file_2.txt',
                      f'{module_path}/test_data/model.json']

    data = [('files', open(path, 'rb')) for path in file_paths]
    data = aiohttp.FormData()
    for path in file_paths:
        data.add_field('files',  open(path, 'rb'))
    return data


def unzip(file_path, target_folder):
    shutil.rmtree(target_folder, ignore_errors=True)
    os.makedirs(target_folder)
    with zipfile.ZipFile(file_path, 'r') as zip:
        zip.extractall(target_folder)


async def condition(condition: Callable, *, timeout: float = 1.0, interval: float = 0.1):
    start = time.time()
    while not condition():
        if time.time() > start + timeout:
            raise TimeoutError(f'condition {condition} took longer than {timeout}s')
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
