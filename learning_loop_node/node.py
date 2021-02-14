from fastapi import FastAPI, Request
import socketio
import asyncio
import functools
from events import Events


class Node(FastAPI):
    def __init__(self, hostname):
        super().__init__()
        self.hostname = hostname
        self.sio = socketio.AsyncClient(
            reconnection_delay=0,
            request_timeout=0.5,
            # logger=True, engineio_logger=True
        )

        self.events = Events(hostname)
        self.sio.register_namespace(self.events)

        @self.on_event("startup")
        async def startup():
            print('startup', flush=True)
            await self.connect()

        @self.on_event("shutdown")
        async def shutdown():
            print('shutting down', flush=True)
            await self.sio.disconnect()

    async def connect(self):
        await self.sio.disconnect()
        print('connecting to Learning Loop', flush=True)

        try:
            await self.sio.connect(f"ws://{self.hostname}", socketio_path="/ws/socket.io")
            print('my sid is', self.sio.sid, flush=True)
        except:
            await asyncio.sleep(0.2)
            await self.connect()
        print('connected to Learning Loop', flush=True)

    def get_weightfile(self, func):
        self.events.get_weightfile = func
