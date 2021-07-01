import asyncio
from datetime import datetime, timedelta
from async_generator import asynccontextmanager
from aiohttp.client_reqrep import ClientResponse
import aiohttp
import os
import werkzeug
from icecream import ic

SERVER_BASE_URL_DEFAULT = 'http://backend'


class AccessToken():
    def __init__(self, token: str, local_expire_date: datetime):
        self.token = token
        self.local_expire_date = local_expire_date

    def is_still_valid(self) -> bool:
        return (self.local_expire_date - timedelta(hours=1)) > datetime.now()


async def token_from_response(response: ClientResponse) -> AccessToken:
    content = await response.json()
    server_time = werkzeug.http.parse_date(response.headers['date'])
    expires_time = datetime.fromtimestamp(content['expires'], tz=server_time.tzinfo)
    # we do not care about the few seconds wich ellapsed during request.
    local_expire_time = datetime.now() + (expires_time - server_time)
    return AccessToken(content['access_token'], local_expire_time)


class LoopHttp():
    def __init__(self) -> None:
        self.base_url: str = os.environ.get('SERVER_BASE_URL', SERVER_BASE_URL_DEFAULT)
        self.username: str = os.environ.get('USERNAME', None)
        self.password: str = os.environ.get('PASSWORD', None)
        self.access_token = None
        self.token_future = None

    async def download_token(self):
        if self.token_future:
            return self.token_future
        self.token_future = asyncio.Future()

        credentials = {
            'username': os.environ.get('USERNAME', None),
            'password': os.environ.get('PASSWORD', None),
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(f'{self.base_url}/api/token', data=credentials) as response:
                assert response.status == 200

                token = await token_from_response(response)
                self.token_future.set_result(token)
                self.token_future = None
                return token

    async def get_headers(self) -> dict:
        if not self.access_token or not self.access_token.is_still_valid():
            self.access_token = await self.download_token()

        headers = {}
        headers['Authorization'] = f'Bearer {self.access_token.token}'
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


instance = LoopHttp()
