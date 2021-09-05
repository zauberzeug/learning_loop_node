import asyncio
from datetime import datetime, timedelta
from async_generator import asynccontextmanager
from aiohttp.client_reqrep import ClientResponse
import aiohttp
import os
import werkzeug
from icecream import ic
import logging


class AccessToken():
    def __init__(self, token: str, local_expire_date: datetime):
        self.token = token
        self.local_expire_date = local_expire_date

    def is_still_valid(self) -> bool:
        return (self.local_expire_date - timedelta(hours=1)) > datetime.now()

    def is_invalid(self) -> bool:
        return not self.is_still_valid


async def token_from_response(response: ClientResponse) -> AccessToken:
    content = await response.json()
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
        ic(self.username, self.password)
        self.organization = os.environ.get('LOOP_ORGANIZATION', None) or os.environ.get('ORGANIZATION', None)
        self.project = os.environ.get('LOOP_PROJECT', None) or os.environ.get('PROJECT', None)

        self.access_token = None
        self.session = None

    async def download_token(self):

        credentials = {
            'username': self.username,
            'password': self.password,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(f'{self.base_url}/api/token', data=credentials) as response:
                logging.debug(f'received token from {self.base_url}')
                assert response.status == 200
                return await token_from_response(response)

    async def ensure_session(self) -> dict:
        '''Create one session for all requests.
        See https://docs.aiohttp.org/en/stable/client_quickstart.html#make-a-request.
        '''

        headers = {}
        if self.username is not None:
            if self.access_token is None or self.access_token.is_invalid():
                self.access_token = await self.download_token()

            headers['Authorization'] = f'Bearer {self.access_token.token}'

        if self.session is None:
            self.session = aiohttp.ClientSession(headers=headers)
        else:
            self.session.headers.update(headers)

    async def get_headers(self):
        await self.ensure_session()
        return self.session.headers

    @asynccontextmanager
    async def get(self, path):
        url = f'{self.base_url}/{path.lstrip("/")}'
        logging.debug(url)
        await self.ensure_session()
        async with self.session.get(url) as response:
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
        await self.ensure_session()
        async with self.session.put(f'{self.base_url}/{path}', data=data) as response:
            yield response

    @asynccontextmanager
    async def post(self, path, data):
        await self.ensure_session()
        async with self.session.post(f'{self.base_url}/{path}', data=data) as response:
            yield response

    @property
    def project_path(self):
        return f'/api/{self.organization}/projects/{self.project}'


loop = Loop()
