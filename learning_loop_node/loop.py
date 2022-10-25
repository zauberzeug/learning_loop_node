import asyncio
from datetime import datetime, timedelta
from async_generator import asynccontextmanager
import aiohttp
import os
import werkzeug
from icecream import ic
import logging
import requests
from . import environment_reader


class AccessToken():
    def __init__(self, token: str, local_expire_date: datetime):
        self.token = token
        self.local_expire_date = local_expire_date

    def is_still_valid(self) -> bool:
        return (self.local_expire_date - timedelta(hours=1)) > datetime.now()

    def is_invalid(self) -> bool:
        return not self.is_still_valid


def token_from_response(response: werkzeug.Response) -> AccessToken:
    content = response.json()
    server_time = werkzeug.http.parse_date(response.headers['date'])
    expires_time = datetime.fromtimestamp(content['expires'], tz=server_time.tzinfo)
    # we do not care about the few seconds wich ellapsed during request.
    local_expire_time = datetime.now() + (expires_time - server_time)
    return AccessToken(content['access_token'], local_expire_time)


class Loop():
    def __init__(self) -> None:
        host = os.environ.get('LOOP_HOST', None) or os.environ.get('HOST', 'learning-loop.ai')
        self.base_url: str = f'http{"s" if host != "backend" else ""}://' + host
        self.username: str = os.environ.get('LOOP_USERNAME', None) or os.environ.get('USERNAME', None)
        self.password: str = os.environ.get('LOOP_PASSWORD', None) or os.environ.get('PASSWORD', None)
        self.organization: str = environment_reader.organization(default='')
        self.project: str = environment_reader.project(default='')
        self.access_token = None
        self.web = requests.Session()

    def download_token(self):

        response = self.web.post(
            (self.base_url + '/api/token').replace('//api', '/api'),
            data={'username': self.username, 'password': self.password}
        )
        response.raise_for_status()
        return token_from_response(response)

    async def create_headers(self) -> dict:
        return await asyncio.get_event_loop().run_in_executor(None, self.get_headers)

    def get_headers(self):
        headers = {}
        if self.username and self.password:
            if self.access_token is None or self.access_token.is_invalid():
                self.access_token = self.download_token()

            headers['Authorization'] = f'Bearer {self.access_token.token}'

        return headers

    @asynccontextmanager
    async def get(self, path):
        url = f'{self.base_url}/{path.lstrip("/")}'
        logging.debug(url)
        async with aiohttp.ClientSession(headers=await self.create_headers()) as session:
            async with session.get(url) as response:
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
        async with aiohttp.ClientSession(headers=await self.create_headers()) as session:
            async with session.put(f'{self.base_url}/{path}', data=data) as response:
                yield response

    @asynccontextmanager
    async def post(self, path, **kwargs):
        async with aiohttp.ClientSession(headers=await self.create_headers()) as session:
            async with session.post(f'{self.base_url}/{path}', **kwargs) as response:
                yield response

    @property
    def project_path(self):
        return f'/api/{self.organization}/projects/{self.project}'


loop = Loop()
