import asyncio
import logging
import time
from typing import Awaitable, Callable, List, Optional

import httpx
from httpx import Cookies, Timeout

from .helpers import environment_reader

logger = logging.getLogger('loop_communication')
logging.getLogger("httpx").setLevel(logging.WARNING)

SLEEP_TIME_ON_429 = 5
MAX_RETRIES_ON_429 = 20


def retry_on_429(func: Callable[..., Awaitable]) -> Callable[..., Awaitable]:
    """Decorator that retries requests that receive a 429 status code."""
    async def wrapper(*args, **kwargs) -> httpx.Response:
        for _ in range(MAX_RETRIES_ON_429):
            response = await func(*args, **kwargs)
            if response.status_code != 429:
                return response

            await asyncio.sleep(SLEEP_TIME_ON_429)

        return response
    return wrapper


class LoopCommunicationException(Exception):
    """Raised when there's an unexpected answer from the learning loop."""


class LoopCommunicator():
    def __init__(self) -> None:
        host: str = environment_reader.host(default='learning-loop.ai')
        self.ssl_cert_path = environment_reader.ssl_certificate_path()
        if self.ssl_cert_path:
            logger.info('Using SSL certificate at %s', self.ssl_cert_path)
        else:
            logger.info('No SSL certificate path set')
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

        logger.info('Loop interface initialized with base_url: %s / user: %s', self.base_url, self.username)

    def websocket_url(self) -> str:
        return f'ws{"s" if "learning-loop.ai" in self.host else ""}://' + self.host

    async def ensure_login(self, relogin=False) -> None:
        """aiohttp client session needs to be created on the event loop"""

        assert not self.async_client.is_closed, 'async client must not be used after shutdown'
        if not self.async_client.cookies.keys() or relogin:
            self.async_client.cookies.clear()
            response = await self.async_client.post('/api/login', data={'username': self.username, 'password': self.password})
            if response.status_code != 200:
                logger.info('Login failed with response: %s', response)
                raise LoopCommunicationException('Login failed with response: ' + str(response))
            self.async_client.cookies.update(response.cookies)

    async def logout(self) -> None:
        """aiohttp client session needs to be created on the event loop"""

        response = await self.async_client.post('/api/logout')
        if response.status_code != 200:
            logger.info('Logout failed with response: %s', response)
            raise LoopCommunicationException('Logout failed with response: ' + str(response))
        self.async_client.cookies.clear()

    def get_cookies(self) -> Cookies:
        return self.async_client.cookies

    async def shutdown(self):
        if self.async_client is not None and not self.async_client.is_closed:
            await self.async_client.aclose()

    async def backend_ready(self, timeout: Optional[int] = None) -> bool:
        """Wait until the backend is ready"""
        start_time = time.time()
        while True:
            try:
                logger.info('Checking if backend is ready')
                response = await self.get('/status', requires_login=False)
                if response.status_code == 200:
                    return True
            except Exception:
                logger.info('backend not ready yet.')
            if timeout is not None and time.time() + 10 - start_time > timeout:
                raise TimeoutError('Backend not ready within timeout')
            await asyncio.sleep(10)

    async def _retry_on_401(self, func: Callable[..., Awaitable[httpx.Response]], *args, **kwargs) -> httpx.Response:
        response = await func(*args, **kwargs)
        if response.status_code == 401:
            await self.ensure_login(relogin=True)
            response = await func(*args, **kwargs)
        return response

    async def get(self, path: str, requires_login: bool = True, api_prefix: str = '/api', timeout: int = 60) -> httpx.Response:
        if requires_login:
            await self.ensure_login()
            return await self._retry_on_401(self._get, path, api_prefix, timeout)
        return await self._get(path, api_prefix)

    @retry_on_429
    async def _get(self, path: str, api_prefix: str, timeout: int = 60) -> httpx.Response:
        return await self.async_client.get(api_prefix+path, timeout=timeout)

    async def put(self, path: str, files: Optional[List[str]] = None, requires_login: bool = True, api_prefix: str = '/api', timeout: int = 60, **kwargs) -> httpx.Response:
        if requires_login:
            await self.ensure_login()
            return await self._retry_on_401(self._put, path, files, api_prefix, timeout, **kwargs)
        return await self._put(path, files, api_prefix, timeout, **kwargs)

    @retry_on_429
    async def _put(self, path: str, files: Optional[List[str]], api_prefix: str, timeout: int = 60, **kwargs) -> httpx.Response:
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
            response = await self.async_client.put(api_prefix+path, files=file_list, timeout=timeout)
        finally:
            for fh in file_handles:
                fh.close()  # Ensure all files are closed

        return response

    async def post(self, path: str, requires_login: bool = True, api_prefix: str = '/api', timeout: int = 60, **kwargs) -> httpx.Response:
        if requires_login:
            await self.ensure_login()
            return await self._retry_on_401(self._post, path, api_prefix, timeout, **kwargs)
        return await self._post(path, api_prefix, **kwargs)

    @retry_on_429
    async def _post(self, path, api_prefix='/api', timeout: int = 60, **kwargs) -> httpx.Response:
        return await self.async_client.post(api_prefix+path, timeout=timeout, **kwargs)

    async def delete(self, path: str, requires_login: bool = True, api_prefix: str = '/api', timeout: int = 60, **kwargs) -> httpx.Response:
        if requires_login:
            await self.ensure_login()
            return await self._retry_on_401(self._delete, path, api_prefix, timeout, **kwargs)
        return await self._delete(path, api_prefix, **kwargs)

    @retry_on_429
    async def _delete(self, path, api_prefix, timeout: int = 60, **kwargs) -> httpx.Response:
        return await self.async_client.delete(api_prefix+path, timeout=timeout, **kwargs)
