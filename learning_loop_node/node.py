import asyncio
import json
import logging
import os
import sys
from abc import abstractmethod
from datetime import datetime
from typing import Optional
from uuid import uuid4

import aiohttp
import socketio
from fastapi import FastAPI
from fastapi_utils.tasks import repeat_every

from learning_loop_node import environment_reader
from learning_loop_node.data_classes.general import Context
from learning_loop_node.globals import GLOBALS

from . import log_conf
from .loop_communication import glc
from .socket_response import ensure_socket_response
from .status import State, Status


class Node(FastAPI):

    def __init__(self, name: str, uuid: Optional[str] = None):
        super().__init__()
        log_conf.init()
        self.log = logging.getLogger()

        host = environment_reader.host(default='learning-loop.ai')
        self.ws_url = f'ws{"s" if "learning-loop.ai" in host else ""}://' + host

        self.name = name
        self.uuid = self.read_or_create_uuid(self.name) if uuid is None else uuid
        self.startup_time = datetime.now()
        self.sio_client: Optional[socketio.AsyncClient] = None
        self.status = Status(id=self.uuid, name=self.name)
        self.register_lifecycle_events()

    def register_lifecycle_events(self):
        @self.on_event("startup")
        async def startup():
            await self.on_startup()

        @self.on_event("shutdown")  # NOTE may only used for developent ?!
        async def shutdown():
            await self.on_shutdown()

        @self.on_event("startup")
        @repeat_every(seconds=10, raise_exceptions=False, wait_first=True)
        async def ensure_connected() -> None:
            await self.on_repeat()

    async def on_startup(self):
        self.log.debug('received "startup" event')
        Node._activate_asyncio_warnings()
        await glc.backend_ready()
        await glc.get_asyncclient()
        await self.create_sio_client()

    async def on_shutdown(self):
        self.log.debug('received "shutdown" event')
        if self.sio_client is not None:
            await self.sio_client.disconnect()

    async def on_repeat(self):
        self.log.debug('received "repeat" event')
        while self.sio_client is None:
            self.log.info('###732 Waiting for sio client to be initialized')
            await asyncio.sleep(1)
        if not self.sio_client.connected:
            self.log.info('###732 Reconnecting to loop via sio')
            await self.connect()
        self.log.info(f'###732 current connection state: {self.sio_client.connected}')

    async def create_sio_client(self):
        cookies = await glc.get_cookies()

        self.sio_client = socketio.AsyncClient(
            reconnection_delay=1,
            request_timeout=0.5,
            http_session=aiohttp.ClientSession(cookies=cookies),  # logger=True, engineio_logger=True
        )

        # pylint: disable=protected-access
        self.sio_client._trigger_event = ensure_socket_response(self.sio_client._trigger_event)
        self.reset_status()

        @self.sio_client.event
        async def connect():
            self.log.debug('received "connect" from loop.')
            self.reset_status()
            state = await self.get_state()
            try:
                await self.update_state(state)
            except:
                self.log.exception('Error sending state.')
                raise

        @self.sio_client.event
        async def disconnect():
            self.log.debug('received "disconnect" from loop.')
            await self.update_state(State.Offline)

    async def connect(self):
        if not self.sio_client:
            self.log.info('###732 sio_client not yet initialized')
            return
        try:
            await self.sio_client.disconnect()
        except Exception:
            pass

        self.log.info(f'(re)connecting to Learning Loop at {self.ws_url}')
        try:
            await self.sio_client.connect(f"{self.ws_url}", headers=self.get_sio_headers(), socketio_path="/ws/socket.io")
            self.log.debug(f'my sid is {self.sio_client.sid}')
            self.log.debug(f"connecting as type {self.get_node_type()}")
            self.log.info(f'connected to Learning Loop at {self.ws_url}')
        except socketio.exceptions.ConnectionError:
            self.log.error(f'socket.io connection error to "{self.ws_url}"')
        except Exception:
            self.log.exception(f'error while connecting to "{self.ws_url}"')

    def get_sio_headers(self) -> dict:
        headers = {}
        headers['organization'] = glc.organization
        headers['project'] = glc.project
        headers['nodeType'] = self.get_node_type()
        return headers

    def read_or_create_uuid(self, identifier: str) -> str:
        identifier = identifier.lower().replace(' ', '_')
        uuids = {}
        file_path = f'{GLOBALS.data_folder}/uuids.json'
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                uuids = json.load(f)

        uuid = uuids.get(identifier, None)
        if not uuid:
            uuid = str(uuid4())
            uuids[identifier] = uuid
            with open(file_path, 'w') as f:
                json.dump(uuids, f)
        return uuid

    def reset_status(self):
        self.status = Status(id=self.uuid, name=self.name)

    async def update_state(self, state: State):
        self.status.state = state
        if self.status.state != State.Offline:
            await self.send_status()

    @abstractmethod
    async def send_status(self):
        pass

    @abstractmethod
    async def get_state(self) -> State:
        pass

    @abstractmethod
    def get_node_type(self):
        pass

    @staticmethod
    def create_project_folder(context: Context) -> str:
        project_folder = f'{GLOBALS.data_folder}/{context.organization}/{context.project}'
        os.makedirs(project_folder, exist_ok=True)
        return project_folder

    @staticmethod
    def _activate_asyncio_warnings() -> None:
        '''Produce warnings for coroutines which take too long on the main loop and hence clog the event loop'''
        try:
            if sys.version_info.major >= 3 and sys.version_info.minor >= 7:  # most
                loop = asyncio.get_running_loop()
            else:
                loop = asyncio.get_event_loop()

            loop.set_debug(True)
            loop.slow_callback_duration = 0.2
            logging.info('activated asyncio warnings')
        except Exception:
            logging.exception('could not activate asyncio warnings')
