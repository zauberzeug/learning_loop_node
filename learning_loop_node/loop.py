from contextlib import asynccontextmanager
from aiohttp.client_reqrep import ClientResponse
import aiohttp
import os


SERVER_BASE_URL_DEFAULT = 'http://backend'


class LoopHttp():
    def __init__(self) -> None:
        self.base_url: str = os.environ.get('SERVER_BASE_URL', SERVER_BASE_URL_DEFAULT)
        self.username: str = os.environ.get('USERNAME', None)
        self.password: str = os.environ.get('PASSWORD', None)

    async def get_token(self):
        credentials = {
            'username': self.username,
            'password': self.password,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(f'{self.base_url}/api/token', data=credentials) as response:
                assert response.status == 200
                content = await response.json()
                return content['access_token']

    async def get_headers(self) -> dict:
        token = await self.get_token()
        headers = {}
        headers['Authorization'] = f'Bearer {token}'
        return headers

    @asynccontextmanager
    async def get(self, path):
        headers = await self.get_headers()
        async with aiohttp.ClientSession() as session:
            async with session.get(f'{self.base_url}/{path}', headers=headers) as response:
                yield response

    @asynccontextmanager
    async def put(self, path, data):
        headers = await self.get_headers()
        async with aiohttp.ClientSession() as session:
            async with session.put(f'{self.base_url}/{path}', headers=headers, data=data) as response:
                yield response

    @asynccontextmanager
    async def post(self, path, data):
        headers = await self.get_headers()
        async with aiohttp.ClientSession() as session:
            async with session.post(f'{self.base_url}/{path}', headers=headers, data=data) as response:
                yield response
