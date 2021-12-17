from glob import glob
import os
from typing import List, Optional
from learning_loop_node.loop import loop
from urllib.parse import urljoin
from requests import Session
import aiohttp
import logging


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


async def assert_upload_model(file_paths: Optional[List[str]] = None, format: str = 'mocked') -> str:
    module_path = os.path.dirname(os.path.realpath(__file__))
    if not file_paths:
        file_paths = [f'{module_path}/test_data/file_1.txt',
                      f'{module_path}/test_data/file_2.txt']
    data = [('files', open(path, 'rb')) for path in file_paths]

    data = aiohttp.FormData()

    for path in file_paths:
        data.add_field('files',  open(path, 'rb'))
    async with loop.post(f'api/zauberzeug/projects/pytest/models/{format}', data) as response:
        if response.status != 200:
            msg = f'unexpected status code {response.status} while posting new model'
            logging.error(msg)
            raise(Exception(msg))
        model = await response.json()
        return model['id']
