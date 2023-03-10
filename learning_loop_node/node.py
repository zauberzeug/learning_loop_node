from learning_loop_node.context import Context
import logging
from learning_loop_node.globals import GLOBALS
from .status import Status, State
from fastapi import FastAPI
import socketio
import asyncio
import asyncio
import os
from icecream import ic
from .loop import loop
from fastapi_utils.tasks import repeat_every
import logging
from uuid import uuid4
from .socket_response import ensure_socket_response
from datetime import datetime
from . import log_conf
import json

log_conf.init()


class Node(FastAPI):
    name: str
    uuid: str

    def __init__(self, name: str, uuid: str = None):
        super().__init__()
        self.log = logging.getLogger()

        host = os.environ.get('LOOP_HOST', None) or os.environ.get('HOST', 'learning-loop.ai')
        self.ws_url = f'ws{"s" if host != "backend" else ""}://' + host

        self.name = name
        self.uuid = self.read_or_create_uuid(self.name) if uuid is None else uuid
        self.startup_time = datetime.now()

        self.sio_client = socketio.AsyncClient(
            reconnection_delay=0,
            request_timeout=0.5,
            # logger=True, engineio_logger=True
        )
        self.sio_client._trigger_event = ensure_socket_response(self.sio_client._trigger_event)
        self.reset()

        @self.sio_client.on('connect')
        async def on_connect():
            logging.debug('received "on_connect" from constructor event.')
            self.reset()
            state = self.get_state()
            try:
                await self.update_state(state)
            except:
                logging.exception(f'Error sending state.')
                raise

        @self.sio_client.on('disconnect')
        async def on_disconnect():
            logging.debug('received "on_disconnect" from constructor event.')
            await self.update_state(State.Offline)

        self.register_lifecycle_events()

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

    def register_lifecycle_events(self):
        @self.on_event("startup")
        async def startup():
            logging.debug('received "startup" event')
            Node._activate_asyncio_warnings()

        @self.on_event("shutdown")
        async def shutdown():
            logging.debug('received "shutdown" event')
            await self.sio_client.disconnect()

        @self.on_event("startup")
        @repeat_every(seconds=10, raise_exceptions=False, wait_first=False)
        async def ensure_connected() -> None:
            logging.info(f'###732 current connection state: {self.sio_client.connected}')
            if not self.sio_client.connected:
                await self.connect()

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
            headers = await self.get_sio_headers()
            await self.sio_client.connect(f"{self.ws_url}", headers=headers, socketio_path="/ws/socket.io")
            logging.debug(f'my sid is {self.sio_client.sid}')
            logging.debug(f"connecting as type {headers['nodeType']}")
            logging.info(f'connected to Learning Loop at {self.ws_url}')
        except socketio.exceptions.ConnectionError as e:
            logging.error(f'socket.io connection error to "{self.ws_url}"')
        except Exception:
            logging.exception(f'error while connecting to "{self.ws_url}"')

    async def update_state(self, state: State):
        self.status.state = state
        if self.status.state != State.Offline:
            await self.send_status()

    async def send_status(self):
        raise Exception("Override this in subclass")

    async def get_sio_headers(self) -> dict:
        headers = await asyncio.get_event_loop().run_in_executor(None, loop.get_headers)
        headers['organization'] = loop.organization
        headers['project'] = loop.project
        headers['nodeType'] = self.get_node_type()
        return headers

    def get_node_type(self):
        classname = self.__class__.__name__
        if classname == 'TrainerNode':
            return 'trainer'
        elif classname == 'DetectorNode':
            return 'detector'
        elif classname == 'ConverterNode':
            return 'converter'
        elif classname == 'AnnotationNode':
            return 'annotation_node'
        else:
            return classname

    @staticmethod
    def create_project_folder(context: Context) -> str:
        project_folder = f'{GLOBALS.data_folder}/{context.organization}/{context.project}'
        os.makedirs(project_folder, exist_ok=True)
        return project_folder

    @staticmethod
    def _activate_asyncio_warnings() -> None:
        '''Produce warnings for coroutines which take too long on the main loop and hence clog the event loop'''
        try:
            import sys
            if sys.version_info.major >= 3 and sys.version_info.minor >= 7:  # most
                loop = asyncio.get_running_loop()
            else:
                loop = asyncio.get_event_loop()

            loop.set_debug(True)
            loop.slow_callback_duration = 0.2
            logging.info('activated asyncio warnings')
        except:
            logging.exception('could not activate asyncio warnings')
