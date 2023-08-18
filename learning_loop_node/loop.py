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
        base_url: str = f'http{"s" if host != "proxy" else ""}://' + host
        logging.info(f'using base_url: {base_url}')
        self.web = WebSession(base_url=base_url)
        self.client_session = None

    async def ensure_login(self):
        # delayed login because the aiohttp client session needs to be created on the event loop
        if not self.web.cookies.keys():
            response = self.web.post('api/login', data={'username': self.username, 'password': self.password})
            if response.status_code != 200:
                self.web.cookies.clear()
                raise Exception('bad response: ' + str(response.content))
            self.web.cookies.update(response.cookies)
        if self.client_session is None or self.client_session.closed:
            self.client_session = aiohttp.ClientSession(base_url=self.web.base_url)
            for cookie in self.web.cookies:
                self.client_session.cookie_jar.update_cookies({cookie.name: cookie.value})

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

    @asynccontextmanager
    async def get(self, path):
        await self.ensure_login()
        async with self.client_session as session:
            path = self.update_path(path)
            async with session.get(path) as response:
                yield response

    async def get_json_async(self, path):
        url = f'{loop.project_path}{path}'
        async with self.get(url) as response:
            if response.status != 200:
                raise Exception('bad response: ' + str(response))
            return await response.json()

    def get_json(self, path):
        return asyncio.get_event_loop().run_until_complete(self.get_json_async(path))

    async def get_data_async(self, path):
        async with self.get(f'{loop.project_path}{path}') as response:
            if response.status != 200:
                raise Exception('bad response: ' + str(response))
            return await response.read()

    def get_data(self, path):
        return asyncio.get_event_loop().run_until_complete(self.get_data_async(path))

    @asynccontextmanager
    async def put(self, path, data):
        await self.ensure_login()
        async with self.client_session as session:
            path = self.update_path(path)
            async with session.put(path, data=data) as response:
                yield response

    async def put_json_async(self, path, json):
        url = f'{loop.project_path}/{path.lstrip("/")}'
        async with self.client_session as session:
            async with session.put(url, json=json) as response:
                if response.status != 200:
                    res = await response.json()
                    raise Exception(f'bad response: {str(response)} \n {res}')
                return await response.json()

    def put_json(self, path, json):
        return asyncio.get_event_loop().run_until_complete(self.put_json_async(path, json))

    @asynccontextmanager
    async def post(self, path, **kwargs):
        await self.ensure_login()
        async with self.client_session as session:
            path = self.update_path(path)
            async with session.post(path, **kwargs) as response:
                yield response

    @property
    def project_path(self):
        return f'/{self.organization}/projects/{self.project}'


loop = Loop()
