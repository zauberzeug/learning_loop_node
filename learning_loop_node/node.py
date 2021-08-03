from learning_loop_node.context import Context
import logging
import aiohttp
from .status import Status, State
from fastapi import FastAPI
import socketio
import asyncio
import asyncio
import os
from icecream import ic
from .loop import loop
from aiohttp.client_exceptions import ClientConnectorError
import logging


class Node(FastAPI):
    name: str
    uuid: str

    def __init__(self, name: str, uuid: str):
        super().__init__()
        host = os.environ.get('HOST', 'learning-loop.ai')
        self.ws_url = f'ws{"s" if host != "backend" else ""}://' + host
        self.organization = os.environ.get('ORGANIZATION', None)
        self.project = os.environ.get('PROJECT', None)

        self.name = name
        self.uuid = uuid

        self.sio_client = socketio.AsyncClient(
            reconnection_delay=0,
            request_timeout=0.5,
            # logger=True, engineio_logger=True
        )
        self.reset()

        @self.sio_client.on('connect')
        async def on_connect():
            logging.debug('received "on_connect" from constructor event.')
            self.reset()
            state = await self.get_state()
            await self.update_state(state)

        @self.sio_client.on('disconnect')
        async def on_disconnect():
            logging.debug('received "on_disconnect" from constructor event.')
            await self.update_state(State.Offline)

        self.register_lifecycle_events()

    def register_lifecycle_events(self):
        @self.on_event("startup")
        async def startup():
            logging.debug('received "startup" event')
            await self.connect()

        @self.on_event("shutdown")
        async def shutdown():
            logging.debug('received "shutdown" event')
            await self.sio_client.disconnect()

    def reset(self):
        self.status = Status(id=self.uuid, name=self.name)
    
    async def get_state(self):
        raise Exception("Override this in subclass")

    async def connect(self):
        try:
            await self.sio_client.disconnect()
        except:
            pass

        logging.info(f'connecting to Learning Loop at {self.ws_url}')
        try:
            headers = await loop.get_headers()
            await self.sio_client.connect(f"{self.ws_url}", headers=headers, socketio_path="/ws/socket.io")
            logging.debug(f'my sid is {self.sio_client.sid}')
            logging.info('connected to Learning Loop')
        except socketio.exceptions.ConnectionError as e:
            logging.error(f'socket.io connection error to "{self.ws_url}"')
            if not ('Already connected' in str(e) or 'Connection refused' in str(e) or 'Unexpected status code' in str(e)):
                await asyncio.sleep(0.5)
                await self.connect()
        except ConnectionRefusedError or ClientConnectorError:
            await asyncio.sleep(0.5)
            await self.connect()
        except Exception:
            logging.error(f'error while connecting to "{self.ws_url}"')
            await asyncio.sleep(2)
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
    def create_project_folder(context:Context) -> str:
        project_folder = f'/data/{context.organization}/{context.project}'
        os.makedirs(project_folder, exist_ok=True)
        return project_folder
