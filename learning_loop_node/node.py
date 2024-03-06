import asyncio
import logging
import os
import sys
from abc import abstractmethod
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional

import aiohttp
import socketio
from fastapi import FastAPI
from socketio import AsyncClient

from .data_classes import Context, NodeState, NodeStatus
from .data_exchanger import DataExchanger
from .globals import GLOBALS
from .helpers import log_conf
from .helpers.misc import ensure_socket_response, read_or_create_uuid
from .loop_communication import LoopCommunicator


class Node(FastAPI):

    def __init__(self, name: str, uuid: Optional[str] = None, needs_login: bool = True):
        """Base class for all nodes. A node is a process that communicates with the zauberzeug learning loop.

        Args:
            name (str): The name of the node. This name is used to generate a uuid.
            uuid (Optional[str]): The uuid of the node. If None, a uuid is generated based on the name 
                and stored in f'{GLOBALS.data_folder}/uuids.json'. 
                From the second run, the uuid is recovered based on the name of the node.
            needs_login (bool): If True, the node will try to login to the learning loop.
        """

        super().__init__(lifespan=self.lifespan)
        log_conf.init()

        self.name = name
        self.uuid = uuid or read_or_create_uuid(self.name)
        self.needs_login = needs_login

        self.log = logging.getLogger()
        self.loop_communicator = LoopCommunicator()
        self.websocket_url = self.loop_communicator.websocket_url()
        self.data_exchanger = DataExchanger(None, self.loop_communicator)

        self.startup_datetime = datetime.now()
        self._sio_client: Optional[AsyncClient] = None
        self.status = NodeStatus(id=self.uuid, name=self.name)

        self.sio_headers = {'organization': self.loop_communicator.organization,
                            'project': self.loop_communicator.project,
                            'nodeType': self.get_node_type()}

        self.repeat_task: Any = None

    @property
    def sio_client(self) -> AsyncClient:
        if self._sio_client is None:
            raise Exception('sio_client not yet initialized')
        return self._sio_client

    def sio_is_initialized(self) -> bool:
        return self._sio_client is not None

    # --------------------------------------------------- APPLICATION LIFECYCLE ---------------------------------------------------
    @asynccontextmanager
    async def lifespan(self, app: FastAPI):

        try:
            self.repeat_task = asyncio.create_task(self.repeat_loop())
            await self._on_startup()
            yield
        finally:
            await self._on_shutdown()
            if self.repeat_task is not None:
                self.repeat_task.cancel()
                try:
                    await self.repeat_task
                except asyncio.CancelledError:
                    pass

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

    async def repeat_loop(self) -> None:
        while True:
            try:
                await self._on_repeat()
            except asyncio.CancelledError:
                return
            except Exception as e:
                self.log.exception(f'error in repeat loop: {e}')
            await asyncio.sleep(5)

    async def _on_repeat(self):
        print('received "repeat" lifecycle-event', flush=True)
        logging.info('received "repeat" lifecycle-event')
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

        self.log.info(f'(re)connecting to Learning Loop at {self.websocket_url}')
        try:
            await self.sio_client.connect(f"{self.websocket_url}", headers=self.sio_headers, socketio_path="/ws/socket.io")
            self.log.info('connected to Learning Loop')
        except socketio.exceptions.ConnectionError:  # type: ignore
            self.log.warning('connection error')
        except Exception:
            self.log.exception(f'error while connecting to "{self.websocket_url}". Exception:')

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
