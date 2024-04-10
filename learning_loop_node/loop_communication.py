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
        self.async_client.cookies.clear()

        logging.info(f'Loop interface initialized with base_url: {self.base_url} / user: {self.username}')

    def websocket_url(self) -> str:
        return f'ws{"s" if "learning-loop.ai" in self.host else ""}://' + self.host

    async def ensure_login(self, relogin=False) -> None:
        """aiohttp client session needs to be created on the event loop"""

        assert not self.async_client.is_closed, 'async client must not be used after shutdown'
        if not self.async_client.cookies.keys() or relogin:
            self.async_client.cookies.clear()
            response = await self.async_client.post('/api/login', data={'username': self.username, 'password': self.password})
            if response.status_code != 200:
                logging.info(f'Login failed with response: {response}')
                raise LoopCommunicationException('Login failed with response: ' + str(response))
            self.async_client.cookies.update(response.cookies)

    async def logout(self) -> None:
        """aiohttp client session needs to be created on the event loop"""

        response = await self.async_client.post('/api/logout')
        if response.status_code != 200:
            logging.info(f'Logout failed with response: {response}')
            raise LoopCommunicationException('Logout failed with response: ' + str(response))
        self.async_client.cookies.clear()

    def get_cookies(self) -> Cookies:
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

        response = await self.async_client.get(api_prefix+path)

        if response.status_code == 401:
            await self.ensure_login(relogin=True)
        return response

    async def put(self, path, files: Optional[List[str]] = None, requires_login=True, api_prefix='/api', **kwargs) -> httpx.Response:
        if requires_login:
            await self.ensure_login()
        if files is None:
            return await self.async_client.put(api_prefix+path, **kwargs)

        file_handles = []
        for f in files:
            try:
                file_handles.append(open(f, 'rb'))
            except FileNotFoundError:
                for fh in file_handles:
                    fh.close()  # Ensure all files are closed
                return httpx.Response(404, content=b'File not found')

        try:
            file_list = [('files', fh) for fh in file_handles]  # Use file handles
            response = await self.async_client.put(api_prefix+path, files=file_list)
        finally:
            for fh in file_handles:
                fh.close()  # Ensure all files are closed

        if response.status_code == 401:
            await self.ensure_login(relogin=True)
        return response

    async def post(self, path, requires_login=True, api_prefix='/api', **kwargs) -> httpx.Response:
        if requires_login:
            await self.ensure_login()
        response = await self.async_client.post(api_prefix+path, **kwargs)

        if response.status_code == 401:
            await self.ensure_login(relogin=True)
        return response

    async def delete(self, path, requires_login=True, api_prefix='/api', **kwargs) -> httpx.Response:
        if requires_login:
            await self.ensure_login()

        response = await self.async_client.delete(api_prefix+path, **kwargs)

        if response.status_code == 401:
            await self.ensure_login(relogin=True)
        return response
