import asyncio
import logging
from typing import Awaitable, Callable, List, Optional

import httpx
from httpx import Cookies, Timeout

from .helpers import environment_reader

logging.basicConfig(level=logging.INFO)


class LoopCommunicationException(Exception):
    """Raised when there's an unexpected answer from the learning loop."""


class LoopCommunicator():
    def __init__(self) -> None:
        host: str = environment_reader.host(default='learning-loop.ai')
        self.ssl_cert_path = environment_reader.ssl_certificate_path()
        if self.ssl_cert_path:
            logging.info('Using SSL certificate at %s', self.ssl_cert_path)
        else:
            logging.info('No SSL certificate path set')
        self.host: str = host
        self.username: str = environment_reader.username()
        self.password: str = environment_reader.password()
        self.organization: str = environment_reader.organization()  # used by mock_detector
        self.project: str = environment_reader.project()  # used by mock_detector
        self.base_url: str = f'http{"s" if "learning-loop.ai" in host else ""}://' + host
        if self.ssl_cert_path:
            self.async_client = httpx.AsyncClient(
                base_url=self.base_url, timeout=Timeout(60.0), verify=self.ssl_cert_path)
        else:
            self.async_client = httpx.AsyncClient(base_url=self.base_url, timeout=Timeout(60.0))
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

    async def retry_on_401(self, func: Callable[..., Awaitable[httpx.Response]], *args, **kwargs) -> httpx.Response:
        response = await func(*args, **kwargs)
        if response.status_code == 401:
            await self.ensure_login(relogin=True)
            response = await func(*args, **kwargs)
        return response

    async def get(self, path: str, requires_login: bool = True, api_prefix: str = '/api') -> httpx.Response:
        if requires_login:
            await self.ensure_login()
            return await self.retry_on_401(self._get, path, api_prefix)
        else:
            return await self._get(path, api_prefix)

    async def _get(self, path: str, api_prefix: str) -> httpx.Response:
        return await self.async_client.get(api_prefix+path)

    async def put(self, path: str, files: Optional[List[str]] = None, requires_login: bool = True, api_prefix: str = '/api', **kwargs) -> httpx.Response:
        if requires_login:
            await self.ensure_login()
            return await self.retry_on_401(self._put, path, files, api_prefix, **kwargs)
        else:
            return await self._put(path, files, api_prefix, **kwargs)

    async def _put(self, path: str, files: Optional[List[str]], api_prefix: str, **kwargs) -> httpx.Response:
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

        return response

    async def post(self, path: str, requires_login: bool = True, api_prefix: str = '/api', **kwargs) -> httpx.Response:
        if requires_login:
            await self.ensure_login()
            return await self.retry_on_401(self._post, path, api_prefix, **kwargs)
        else:
            return await self._post(path, api_prefix, **kwargs)

    async def _post(self, path, api_prefix='/api', **kwargs) -> httpx.Response:
        return await self.async_client.post(api_prefix+path, **kwargs)

    async def delete(self, path: str, requires_login: bool = True, api_prefix: str = '/api', **kwargs) -> httpx.Response:
        if requires_login:
            await self.ensure_login()
            return await self.retry_on_401(self._delete, path, api_prefix, **kwargs)
        else:
            return await self._delete(path, api_prefix, **kwargs)

    async def _delete(self, path, api_prefix, **kwargs) -> httpx.Response:
        return await self.async_client.delete(api_prefix+path, **kwargs)
