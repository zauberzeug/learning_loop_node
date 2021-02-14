from fastapi import FastAPI, Request
import socketio
import asyncio


class Node(FastAPI):
    def __init__(self, hostname):
        super().__init__()
        self.hostname = hostname
        self.sio = socketio.AsyncClient(
            reconnection_delay=0,
            request_timeout=0.5,
            # logger=True, engineio_logger=True
        )

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
