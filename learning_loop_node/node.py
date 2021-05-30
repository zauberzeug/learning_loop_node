from .status import Status, State
from fastapi import FastAPI
import socketio
import asyncio
import asyncio
import requests
import os
from icecream import ic

SERVER_BASE_URL_DEFAULT = 'http://backend'
WEBSOCKET_BASE_URL_DEFAULT = 'ws://backend'
BASE_PROJECT = 'demo'
BASE_ORGANIZATION = 'zauberzeug'


class Node(FastAPI):
    name: str
    uuid: str

    def __init__(self, name: str, uuid: str):
        super().__init__()
        self.url = os.environ.get('SERVER_BASE_URL', SERVER_BASE_URL_DEFAULT)
        self.ws_url = os.environ.get('WEBSOCKET_BASE_URL', WEBSOCKET_BASE_URL_DEFAULT)
        self.username = os.environ.get('USERNAME', None)
        self.password = os.environ.get('PASSWORD', None)
        self.project = os.environ.get('PROJECT', BASE_PROJECT)
        self.organization = os.environ.get('ORGANIZATION', BASE_ORGANIZATION)
        self.db_user = os.environ.get('DBNAME', None)
        self.db_pw = os.environ.get('DBPW', None)
        self.headers = {}

        self.name = name
        self.uuid = uuid

        if self.username:
            import base64
            self.headers["Authorization"] = "Basic " + \
                base64.b64encode(f"{self.username}:{self.password}".encode()).decode()

        self.sio = socketio.AsyncClient(
            reconnection_delay=0,
            request_timeout=0.5,
            # logger=True, engineio_logger=True
        )
        self.reset()

        @self.sio.on('connect')
        async def on_connect():
            ic('received "on_connect" from constructor event.')
            self.reset()
            await self.update_state(State.Idle)

        @self.sio.on('disconnect')
        async def on_disconnect():
            ic('received "on_disconnect" from constructor event.')
            await self.update_state(State.Offline)

        self.register_lifecycle_events()

    def register_lifecycle_events(self):
        @self.on_event("startup")
        async def startup():
            print('received "startup" event', flush=True)
            await self.connect()

        @self.on_event("shutdown")
        async def shutdown():
            print('received "shutdown" event', flush=True)
            await self.sio.disconnect()

    def reset(self):
        self.status = Status(id=self.uuid, name=self.name)

    async def connect(self):
        try:
            await self.sio.disconnect()
        except:
            pass

        print('connecting to Learning Loop', flush=True)
        try:
            await self.sio.connect(f"{self.ws_url}", auth={'username': self.db_user, 'password': self.db_pw}, headers=self.headers, socketio_path="/ws/socket.io")
            print('my sid is', self.sio.sid, flush=True)
            print('connected to Learning Loop', flush=True)
        except socketio.exceptions.ConnectionError as e:
            ic(e)
            if 'Already connected' in str(e):
                print('we are already connected')
            else:
                await asyncio.sleep(0.2)
                await self.connect()

    async def update_state(self, state: State):
        self.status.state = state
        if self.status.state != State.Offline:
            await self.send_status()

    async def update_status(self, new_status: Status):
        self.status.id = new_status.id
        self.status.name = new_status.name
        self.status.uptime = new_status.uptime
        self.status.latest_error = new_status.latest_error

        if self.status.state != State.Offline:
            self.status.state = State.Idle
        await self.send_status()

    async def send_status(self):
        raise Exception("Override this in subclass")

    @staticmethod
    def create_project_folder(organization: str, project: str) -> str:
        project_folder = f'/data/{organization}/{project}'
        os.makedirs(project_folder, exist_ok=True)
        return project_folder
