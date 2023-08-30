import asyncio
import logging
import os
import shutil
import time
import zipfile
from glob import glob
from typing import Callable, List, Optional
from urllib.parse import urljoin

import aiohttp
import requests
from icecream import ic
from requests import Session

from learning_loop_node.loop_communication import glc


class LiveServerSession(Session):
    """https://stackoverflow.com/a/51026159/364388"""

    def __init__(self, *args, **kwargs):
        super(LiveServerSession, self).__init__(*args, **kwargs)
        self.prefix_url = glc.web.base_url
        data = {
            'username': os.environ.get('LOOP_USERNAME', None),
            'password': os.environ.get('LOOP_PASSWORD', None),
        }
        self.cookies = requests.post(f'{self.prefix_url}/api/login', data=data).cookies

    def request(self, method, url, *args, **kwargs):
        url = 'api/' + url
        url = urljoin(self.prefix_url, url)
        return super(LiveServerSession, self).request(method, url, cookies=self.cookies, *args, **kwargs)


def get_files_in_folder(folder: str):
    files = [entry for entry in glob(f'{folder}/**/*', recursive=True) if os.path.isfile(entry)]
    files.sort()
    return files


async def get_latest_model_id() -> str:
    response = await glc.get(f'/zauberzeug/projects/pytest/trainings')
    assert response.status_code == 200
    trainings = response.json()
    return trainings['charts'][0]['data'][0]['model_id']


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
