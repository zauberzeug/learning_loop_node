from fastapi import FastAPI, Request
import socketio
import asyncio
import functools
import requests


class Node(FastAPI):
    def __init__(self, hostname):
        super().__init__()
        self.hostname = hostname
        self.sio = socketio.AsyncClient(
            reconnection_delay=0,
            request_timeout=0.5,
            # logger=True, engineio_logger=True
        )

        @self.on_event("startup")
        async def startup():
            print('startup', flush=True)
            await self.connect()

        @self.on_event("shutdown")
        async def shutdown():
            print('shutting down', flush=True)
            await self.sio.disconnect()

        @self.sio.on('save')
        async def save(model):
            print('---- saving model', model['id'], flush=True)
            context = model['context']
            response = requests.put(
                f'http://{hostname}/api/{context["organization"]}/projects/{context["project"]}/models/{model["id"]}/file',
                files={'data': self._get_weightfile(model)}
            )
            if response.status_code == 200:
                return True
            else:
                return response.json()['detail']

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
        self._get_weightfile = func
