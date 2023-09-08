import asyncio
import logging
from typing import List, Optional

import httpx
from httpx import Timeout

from . import environment_reader

# set log level to info
logging.basicConfig(level=logging.INFO)


class LoopCommunicationException(Exception):
    """Raised when there's an unexpected answer from the learning loop."""


class LoopCommunicator():
    def __init__(self) -> None:
        host: str = environment_reader.host(default='learning-loop.ai')
        self.username: str = environment_reader.username()
        self.password: str = environment_reader.password()
        self.organization: str = environment_reader.organization()
        self.project: str = environment_reader.project()
        self.base_url: str = f'http{"s" if "learning-loop.ai" in host else ""}://' + host
        self._async_client: Optional[httpx.AsyncClient] = None

        logging.info(f'Loop interface initialized with base_url: {self.base_url}')

    @property
    def project_path(self):
        return f'/{self.organization}/projects/{self.project}'

    async def get_asyncclient(self, requires_login=True) -> httpx.AsyncClient:
        """aiohttp client session needs to be created on the event loop"""

        if self._async_client is None or self._async_client.is_closed:
            logging.info(f'Creating new async client for {self.base_url}')
            self._async_client = httpx.AsyncClient(base_url=self.base_url, timeout=Timeout(60.0))

        if requires_login and not self._async_client.cookies.keys():
            response = await self._async_client.post('/api/login', data={'username': self.username, 'password': self.password})
            if response.status_code != 200:
                self._async_client.cookies.clear()
                logging.info(f'Login failed with response: {response}')
                logging.info(f'username: {self.username}')
                logging.info(f'password: {self.password}')
                raise LoopCommunicationException('Login failed with response: ' + str(response))
            self._async_client.cookies.update(response.cookies)

        return self._async_client

    async def get_cookies(self):
        ac = await self.get_asyncclient()
        return ac.cookies

    async def shutdown(self):
        if self._async_client is not None and not self._async_client.is_closed:
            await self._async_client.aclose()
        self._async_client = None

    async def backend_ready(self) -> bool:
        """Wait until the backend is ready"""
        while True:
            try:
                logging.info('Checking if backend is ready')
                response = await self.get('/status')
                if response.status_code == 200:
                    return True
            except Exception as e:
                logging.info(f'backend not ready: {e}')
            await asyncio.sleep(3)

    async def get(self, path, requires_login=True, api_prefix='/api') -> httpx.Response:
        ac = await self.get_asyncclient(requires_login=requires_login)
        return await ac.get(api_prefix+path)

    async def put(self, path, files: List[str], requires_login=True, api_prefix='/api') -> httpx.Response:
        ac = await self.get_asyncclient(requires_login=requires_login)
        file_list = [('files', open(f, 'rb')) for f in files]  # TODO: does this properly close the files after upload?
        return await ac.put(api_prefix+path, files=file_list)

    async def post(self, path, requires_login=True, api_prefix='/api', **kwargs) -> httpx.Response:
        ac = await self.get_asyncclient(requires_login=requires_login)
        return await ac.post(api_prefix+path, **kwargs)

    async def delete(self, path, requires_login=True, api_prefix='/api', **kwargs) -> httpx.Response:
        ac = await self.get_asyncclient(requires_login=requires_login)
        return await ac.delete(api_prefix+path, **kwargs)

    # --------------------------------- only used by example/novelty_score_updater ---------------------------------

    def get_json(self, path):
        return asyncio.get_event_loop().run_until_complete(self._get_json_async(path))

    async def _get_json_async(self, path):
        url = f'{self.project_path}{path}'
        response = await self.get(url)
        if response.status_code != 200:
            raise LoopCommunicationException('bad response: ' + str(response))
        return response.json()

    def put_json(self, path, json):
        return asyncio.get_event_loop().run_until_complete(self._put_json_async(path, json))

    async def _put_json_async(self, path, json) -> dict:
        ac = await self.get_asyncclient()
        url = f'{self.project_path}/{path.lstrip("/")}'
        response = await ac.put(url, json=json)
        if response.status_code != 200:
            raise LoopCommunicationException(f'bad response: {str(response)} \n {response.json()}')
        return response.json()

    # --------------------------------- unused?! ---------------------------------

    def get_data(self, path):
        return asyncio.get_event_loop().run_until_complete(self._get_data_async(path))

    async def _get_data_async(self, path) -> bytes:
        response = await self.get(f'{self.project_path}{path}')
        if response.status_code != 200:
            raise LoopCommunicationException('bad response: ' + str(response))
        return response.content
