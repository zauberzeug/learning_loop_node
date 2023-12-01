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
from socketio import AsyncClient

from .data_classes import Context, NodeState, NodeStatus
from .data_exchanger import DataExchanger
from .globals import GLOBALS
from .helpers import environment_reader, log_conf
from .helpers.misc import ensure_socket_response
from .loop_communication import LoopCommunicator


class Node(FastAPI):

    def __init__(self, name: str, uuid: Optional[str] = None):
        """Base class for all nodes. A node is a process that communicates with the zauberzeug learning loop.

        Args:
            name (str): The name of the node. This name is used to generate a uuid.
            uuid (Optional[str]): The uuid of the node. If None, a uuid is generated based on the name 
                and stored in f'{GLOBALS.data_folder}/uuids.json'. 
                From the second run, the uuid is recovered based on the name of the node. Defaults to None.
        """

        super().__init__()
        log_conf.init()

        self.log = logging.getLogger()
        self.loop_communicator = LoopCommunicator()
        self.data_exchanger = DataExchanger(None, self.loop_communicator)

        host = environment_reader.host(default='learning-loop.ai')
        self.ws_url = f'ws{"s" if "learning-loop.ai" in host else ""}://' + host

        self.name = name
        self.uuid = self.read_or_create_uuid(self.name) if uuid is None else uuid
        self.startup_time = datetime.now()
        self._sio_client: Optional[AsyncClient] = None
        self.status = NodeStatus(id=self.uuid, name=self.name)
        # NOTE this is can be set to False for Nodes which do not need to authenticate with the backend (like the DetectorNode)
        self.needs_login = True
        self._setup_sio_headers()
        self._register_lifecycle_events()

    @property
    def sio_client(self) -> AsyncClient:
        if self._sio_client is None:
            raise Exception('sio_client not yet initialized')
        return self._sio_client

    def sio_is_initialized(self) -> bool:
        return self._sio_client is not None

    # --------------------------------------------------- INIT ---------------------------------------------------

    def read_or_create_uuid(self, identifier: str) -> str:
        identifier = identifier.lower().replace(' ', '_')
        uuids = {}
        os.makedirs(GLOBALS.data_folder, exist_ok=True)
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

    def _setup_sio_headers(self) -> None:
        self.sio_headers = {'organization': self.loop_communicator.organization,
                            'project': self.loop_communicator.project,
                            'nodeType': self.get_node_type()}

    # --------------------------------------------------- APPLICATION LIFECYCLE ---------------------------------------------------

    def _register_lifecycle_events(self):
        @self.on_event("startup")
        async def startup():
            await self._on_startup()

        @self.on_event("shutdown")  # NOTE only used for developent ?!
        async def shutdown():
            await self._on_shutdown()

        @self.on_event("startup")
        @repeat_every(seconds=5, raise_exceptions=False, wait_first=False)
        async def ensure_connected() -> None:
            await self._on_repeat()

    async def _on_startup(self):
        self.log.info('received "startup" lifecycle-event')
        Node._activate_asyncio_warnings()
        if self.needs_login:
            await self.loop_communicator.backend_ready()
            self.log.info('ensuring login')
            await self.loop_communicator.ensure_login()
        self.log.info('create sio client')
        await self.create_sio_client()
        self.log.info('done')
        await self.on_startup()

    async def _on_shutdown(self):
        self.log.info('received "shutdown" lifecycle-event')
        await self.loop_communicator.shutdown()
        if self._sio_client is not None:
            await self._sio_client.disconnect()
        self.log.info('successfully disconnected from loop.')
        await self.on_shutdown()

    async def _on_repeat(self):
        while not self.sio_is_initialized():
            self.log.info('Waiting for sio client to be initialized')
            await asyncio.sleep(1)
        if not self.sio_client.connected:
            self.log.info('Reconnecting to loop via sio')
            await self.connect_sio()
            if not self.sio_client.connected:
                self.log.warning('Could not connect to loop via sio')
                return
        await self.on_repeat()

    # --------------------------------------------------- SOCKET.IO ---------------------------------------------------

    async def create_sio_client(self):
        cookies = await self.loop_communicator.get_cookies()
        self._sio_client = AsyncClient(request_timeout=20, http_session=aiohttp.ClientSession(cookies=cookies))

        # pylint: disable=protected-access
        self.sio_client._trigger_event = ensure_socket_response(self.sio_client._trigger_event)

        @self._sio_client.event
        async def connect():
            self.log.info('received "connect" via sio from loop.')
            self.status = NodeStatus(id=self.uuid, name=self.name)
            state = await self.get_state()
            try:
                await self._update_send_state(state)
            except:
                self.log.exception('Error sending state. Exception:')
                raise

        @self._sio_client.event
        async def disconnect():
            self.log.info('received "disconnect" via sio from loop.')
            await self._update_send_state(NodeState.Offline)

        @self._sio_client.event
        async def restart():
            self.log.info('received "restart" via sio from loop.')
            self.restart()

        self.register_sio_events(self._sio_client)

    async def connect_sio(self):
        if not self.sio_is_initialized():
            self.log.warning('sio client not yet initialized')
            return
        try:
            await self.sio_client.disconnect()
        except Exception:
            pass

        self.log.info(f'(re)connecting to Learning Loop at {self.ws_url}')
        try:
            await self.sio_client.connect(f"{self.ws_url}", headers=self.sio_headers, socketio_path="/ws/socket.io")
            self.log.info('connected to Learning Loop')
        except socketio.exceptions.ConnectionError:  # type: ignore
            self.log.warning('connection error')
        except Exception:
            self.log.exception(f'error while connecting to "{self.ws_url}". Exception:')

    async def _update_send_state(self, state: NodeState):
        self.status.state = state
        if self.status.state != NodeState.Offline:
            await self.send_status()

    # --------------------------------------------------- ABSTRACT METHODS ---------------------------------------------------

    @abstractmethod
    def register_sio_events(self, sio_client: AsyncClient):
        """Register socket.io events for the communication with the learning loop.
        The events: connect and disconnect are already registered and should not be overwritten."""

    @abstractmethod
    async def send_status(self):
        """Send the current status to the learning loop.
        Note that currently this method is also used to react to the response of the learning loop."""

    @abstractmethod
    async def get_state(self) -> NodeState:
        """Return the current state of the node."""

    @abstractmethod
    def get_node_type(self):
        pass

    @abstractmethod
    async def on_startup(self):
        """This method is called when the node is started."""

    @abstractmethod
    async def on_shutdown(self):
        """This method is called when the node is shut down."""

    @abstractmethod
    async def on_repeat(self):
        """This method is called every 10 seconds."""
    # --------------------------------------------------- SHARED FUNCTIONS ---------------------------------------------------

    def restart(self):
        """Restart the node."""
        self.log.info('restarting node')
        sys.exit(0)

    # --------------------------------------------------- HELPER ---------------------------------------------------

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
            logging.exception('could not activate asyncio warnings. Exception:')
