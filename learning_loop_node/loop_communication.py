import asyncio
import logging
from typing import List, Optional

import httpx
from httpx import Cookies, Timeout

from .helpers import environment_reader

logging.basicConfig(level=logging.INFO)


class LoopCommunicationException(Exception):
    """Raised when there's an unexpected answer from the learning loop."""


class LoopCommunicator():
    def __init__(self) -> None:
        host: str = environment_reader.host(default='learning-loop.ai')
        self.host: str = host
        self.username: str = environment_reader.username()
        self.password: str = environment_reader.password()
        self.organization: str = environment_reader.organization()  # used by mock_detector
        self.project: str = environment_reader.project()  # used by mock_detector
        self.base_url: str = f'http{"s" if "learning-loop.ai" in host else ""}://' + host
        self.async_client: httpx.AsyncClient = httpx.AsyncClient(base_url=self.base_url, timeout=Timeout(60.0))

        logging.info(f'Loop interface initialized with base_url: {self.base_url} / user: {self.username}')

    # @property
    # def project_path(self):  # TODO: remove?
    #     return f'/{self.organization}/projects/{self.project}'

    async def ensure_login(self) -> None:
        """aiohttp client session needs to be created on the event loop"""

        assert not self.async_client.is_closed, 'async client must not be used after shutdown'
        if not self.async_client.cookies.keys():
            response = await self.async_client.post('/api/login', data={'username': self.username, 'password': self.password})
            if response.status_code != 200:
                self.async_client.cookies.clear()
                logging.info(f'Login failed with response: {response}')
                raise LoopCommunicationException('Login failed with response: ' + str(response))
            self.async_client.cookies.update(response.cookies)

    async def logout(self) -> None:
        """aiohttp client session needs to be created on the event loop"""

        response = await self.async_client.post('/api/logout')
        if response.status_code != 200:
            logging.info(f'Logout failed with response: {response}')
            raise LoopCommunicationException('Logout failed with response: ' + str(response))

    async def get_cookies(self) -> Cookies:
        return self.async_client.cookies

    async def shutdown(self):
        if self.async_client is not None and not self.async_client.is_closed:
            await self.async_client.aclose()

    async def backend_ready(self) -> bool:
        """Wait until the backend is ready"""
        while True:
            try:
                logging.info('Checking if backend is ready')
                response = await self.get('/status', requires_login=False)
                if response.status_code == 200:
                    return True
            except Exception as e:
                logging.info(f'backend not ready: {e}')
            await asyncio.sleep(10)

    async def get(self, path: str, requires_login: bool = True, api_prefix: str = '/api') -> httpx.Response:
        if requires_login:
            await self.ensure_login()
        return await self.async_client.get(api_prefix+path)

    async def put(self, path, files: Optional[List[str]]=None, requires_login=True, api_prefix='/api', **kwargs) -> httpx.Response:
        if requires_login:
            await self.ensure_login()
        if files is None:
            return await self.async_client.put(api_prefix+path, **kwargs)
        
        file_list = [('files', open(f, 'rb')) for f in files]  # TODO: does this properly close the files after upload?
        return await self.async_client.put(api_prefix+path, files=file_list)

    async def post(self, path, requires_login=True, api_prefix='/api', **kwargs) -> httpx.Response:
        if requires_login:
            await self.ensure_login()
        return await self.async_client.post(api_prefix+path, **kwargs)

    async def delete(self, path, requires_login=True, api_prefix='/api', **kwargs) -> httpx.Response:
        if requires_login:
            await self.ensure_login()
        return await self.async_client.delete(api_prefix+path, **kwargs)

    # --------------------------------- unused?! --------------------------------- #TODO remove?

    # def get_data(self, path):
    #     return asyncio.get_event_loop().run_until_complete(self._get_data_async(path))

    # async def _get_data_async(self, path) -> bytes:
    #     response = await self.get(f'{self.project_path}{path}')
    #     if response.status_code != 200:
    #         raise LoopCommunicationException('bad response: ' + str(response))
    #     return response.content
