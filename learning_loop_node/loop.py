import asyncio
from datetime import datetime, timedelta
from async_generator import asynccontextmanager
import aiohttp
import os
from icecream import ic
import logging
from . import environment_reader
from requests import Session
from urllib.parse import urljoin
import httpx
from typing import List
from httpx import Timeout


class WebSession(Session):
    def __init__(self, base_url=None):
        super().__init__()
        self.base_url = base_url

    def request(self, method, url, *args, **kwargs):
        joined_url = urljoin(self.base_url, url)
        return super().request(method, joined_url, *args, **kwargs)


class Loop():
    def __init__(self) -> None:
        host = os.environ.get('LOOP_HOST', None) or os.environ.get('HOST', 'learning-loop.ai')
        self.username: str = os.environ.get('LOOP_USERNAME', None) or os.environ.get('USERNAME', None)
        self.password: str = os.environ.get('LOOP_PASSWORD', None) or os.environ.get('PASSWORD', None)
        self.organization: str = environment_reader.organization(default='')
        self.project: str = environment_reader.project(default='')
        base_url: str = f'http{"s" if "learning-loop.ai" in host else ""}://' + host
        logging.info(f'using base_url: {base_url}')
        self.web = WebSession(base_url=base_url)
        self.async_client: httpx.AsyncClient = None

    async def ensure_login(self):
        # delayed login because the aiohttp client session needs to be created on the event loop
        if not self.web.cookies.keys():
            response = self.web.post('api/login', data={'username': self.username, 'password': self.password})
            if response.status_code != 200:
                self.web.cookies.clear()
                raise Exception('bad response: ' + str(response.content))
            self.web.cookies.update(response.cookies)
        if self.async_client is None or self.async_client.is_closed:
            self.async_client = httpx.AsyncClient(
                base_url=self.web.base_url, timeout=Timeout(timeout=60.0))
            for cookie in self.web.cookies:
                self.async_client.cookies.update({cookie.name: cookie.value})

    async def create_headers(self) -> dict:
        return await asyncio.get_event_loop().run_in_executor(None, self.get_headers)

    async def backend_ready(self) -> bool:
        while True:
            try:
                logging.info('checking if backend is ready')
                response = self.web.get('/api/status')
                if response.status_code == 200:
                    return True
            except Exception as e:
                logging.info(f'backend not ready: {e}')
            await asyncio.sleep(2)

    def update_path(self, path: str) -> str:
        return f'/api{path}'

    async def get(self, path) -> httpx.Response:
        await self.ensure_login()
        path = self.update_path(path)
        return await self.async_client.get(path)

    async def get_json_async(self, path):
        url = f'{loop.project_path}{path}'
        response = await self.get(url)
        if response.status_code != 200:
            raise Exception('bad response: ' + str(response))
        return response.json()

    def get_json(self, path):
        return asyncio.get_event_loop().run_until_complete(self.get_json_async(path))

    async def get_data_async(self, path) -> bytes:
        response = await self.get(f'{loop.project_path}{path}')
        if response.status_code != 200:
            raise Exception('bad response: ' + str(response))
        return response.content

    def get_data(self, path):
        return asyncio.get_event_loop().run_until_complete(self.get_data_async(path))

    async def put(self, path, files: List[str]) -> httpx.Response:
        file_list = [('files', open(f, 'rb')) for f in files]
        ic(file_list)
        await self.ensure_login()
        path = self.update_path(path)
        return await self.async_client.put(path, files=file_list)

    async def put_json_async(self, path, json) -> dict:
        url = f'{loop.project_path}/{path.lstrip("/")}'
        response = await self.async_client.put(url, json=json)
        if response.status_code != 200:
            res = response.json()
            raise Exception(f'bad response: {str(response)} \n {res}')
        return response.json()

    def put_json(self, path, json):
        return asyncio.get_event_loop().run_until_complete(self.put_json_async(path, json))

    async def post(self, path, **kwargs) -> httpx.Response:
        await self.ensure_login()
        path = self.update_path(path)
        return await self.async_client.post(path, **kwargs)

    @property
    def project_path(self):
        return f'/{self.organization}/projects/{self.project}'


loop = Loop()
