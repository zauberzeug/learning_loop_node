import asyncio
import logging
import ssl
import sys
from abc import abstractmethod
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional

import aiohttp
from aiohttp import TCPConnector
from fastapi import FastAPI
from socketio import AsyncClient

from .data_classes import NodeStatus
from .data_exchanger import DataExchanger
from .helpers import log_conf
from .helpers.misc import ensure_socket_response, read_or_create_uuid
from .loop_communication import LoopCommunicator
from .rest import router


class NodeConnectionError(Exception):
    pass


class Node(FastAPI):

    def __init__(self, name: str, uuid: Optional[str] = None, node_type: str = 'node', needs_login: bool = True):
        """Base class for all nodes. A node is a process that communicates with the zauberzeug learning loop.
        This class provides the basic functionality to connect to the learning loop via socket.io and to exchange data.

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

        self.log = logging.getLogger('Node')
        self.init_loop_communicator()
        self.data_exchanger = DataExchanger(None, self.loop_communicator)

        self.startup_datetime = datetime.now()
        self._sio_client: Optional[AsyncClient] = None
        self.status = NodeStatus(id=self.uuid, name=self.name)

        self.sio_headers = {'organization': self.loop_communicator.organization,
                            'project': self.loop_communicator.project,
                            'nodeType': node_type}

        self.repeat_task: Any = None
        self.socket_connection_broken = False
        self._skip_repeat_loop = False

        self.include_router(router)

        self.CONNECTED_TO_LOOP = asyncio.Event()
        self.DISCONNECTED_FROM_LOOP = asyncio.Event()

        self.repeat_loop_lock = asyncio.Lock()

    def init_loop_communicator(self):
        self.loop_communicator = LoopCommunicator()
        self.websocket_url = self.loop_communicator.websocket_url()

    @property
    def sio_client(self) -> AsyncClient:
        if self._sio_client is None:
            raise Exception('sio_client not yet initialized')
        return self._sio_client

    # --------------------------------------------------- APPLICATION LIFECYCLE ---------------------------------------------------
    @asynccontextmanager
    async def lifespan(self, app: FastAPI):  # pylint: disable=unused-argument
        try:
            try:
                await self._on_startup()
            except Exception:
                self.log.exception('Fatal error during startup: %s')
            self.repeat_task = asyncio.create_task(self.repeat_loop())
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
        try:
            await self.reconnect_to_loop()
        except Exception:
            self.log.warning('Could not establish sio connection to loop during startup')
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
            if self._skip_repeat_loop:
                self.log.debug('node is muted, skipping repeat loop')
                await asyncio.sleep(1)
                continue
            try:
                async with self.repeat_loop_lock:
                    await self._ensure_sio_connection()
                    await self.on_repeat()
            except asyncio.CancelledError:
                return
            except Exception:
                self.log.exception('error in repeat loop')

            await asyncio.sleep(5)

    async def _ensure_sio_connection(self):
        if self.socket_connection_broken or self._sio_client is None or not self.sio_client.connected:
            self.log.info('Reconnecting to loop via sio due to %s',
                          'broken connection' if self.socket_connection_broken else 'no connection')
            await self.reconnect_to_loop()

    async def reconnect_to_loop(self):
        """Initialize the loop communicator, log in if needed and reconnect to the loop via socket.io."""
        self.init_loop_communicator()
        if self.needs_login:
            await self.loop_communicator.ensure_login(relogin=True)
        try:
            await self._reconnect_socketio()
        except NodeConnectionError:
            self.log.exception('Could not reset sio connection to loop')
            self.socket_connection_broken = True
            raise

        self.socket_connection_broken = False

    def set_skip_repeat_loop(self, value: bool):
        self._skip_repeat_loop = value
        self.log.info('node is muted: %s', value)

    # --------------------------------------------------- SOCKET.IO ---------------------------------------------------

    async def _reconnect_socketio(self):
        """Create a socket.io client, connect it to the learning loop and register its events.
        The current client is disconnected and deleted if it already exists."""

        self.log.debug('-------------- Connecting to loop via socket.io -------------------')
        self.log.debug('HTTP Cookies: %s\n', self.loop_communicator.get_cookies())

        if self._sio_client is not None:
            try:
                await self.sio_client.disconnect()
                self.log.info('disconnected from loop via sio')
                # NOTE: without waiting for the disconnect event, we might disconnect the next connection too early
                await asyncio.wait_for(self.DISCONNECTED_FROM_LOOP.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.log.warning(
                    'Did not receive disconnect event from loop within 5 seconds.\nContinuing with new connection...')
            except Exception as e:
                self.log.warning('Could not disconnect from loop via sio: %s.\nIgnoring...', e)
            self._sio_client = None

        connector = None
        if self.loop_communicator.ssl_cert_path:
            logging.info('SIO using SSL certificate path: %s', self.loop_communicator.ssl_cert_path)
            ssl_context = ssl.create_default_context(cafile=self.loop_communicator.ssl_cert_path)
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_REQUIRED
            connector = TCPConnector(ssl=ssl_context)

        self._sio_client = AsyncClient(request_timeout=20, http_session=aiohttp.ClientSession(
            cookies=self.loop_communicator.get_cookies(), connector=connector))

        # pylint: disable=protected-access
        self._sio_client._trigger_event = ensure_socket_response(self._sio_client._trigger_event)

        @self._sio_client.event
        async def connect():
            self.log.info('received "connect" via sio from loop.')
            self.CONNECTED_TO_LOOP.set()
            self.DISCONNECTED_FROM_LOOP.clear()

        @self._sio_client.event
        async def disconnect():
            self.log.info('received "disconnect" via sio from loop.')
            self.DISCONNECTED_FROM_LOOP.set()
            self.CONNECTED_TO_LOOP.clear()

        @self._sio_client.event
        async def restart():
            self.log.info('received "restart" via sio from loop -> restarting node.')
            sys.exit(0)

        self.register_sio_events(self._sio_client)
        try:
            await self._sio_client.connect(f"{self.websocket_url}", headers=self.sio_headers, socketio_path="/ws/socket.io")
        except Exception as e:
            self.log.exception('Could not connect socketio client to loop')
            raise NodeConnectionError('Could not connect socketio client to loop') from e

        if not self._sio_client.connected:
            self.log.exception('Could not connect socketio client to loop')
            raise NodeConnectionError('Could not connect socketio client to loop')

    # --------------------------------------------------- ABSTRACT METHODS ---------------------------------------------------

    @abstractmethod
    async def on_startup(self):
        """This method is called when the node is started.
        Note: In this method the sio connection is not yet established!"""

    @abstractmethod
    async def on_shutdown(self):
        """This method is called when the node is shut down."""

    @abstractmethod
    async def on_repeat(self):
        """This method is called every 10 seconds."""

    @abstractmethod
    def register_sio_events(self, sio_client: AsyncClient):
        """Register (additional) socket.io events for the communication with the learning loop.
        The events: connect, disconnect and restart are already registered and should not be overwritten."""
