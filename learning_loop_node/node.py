from fastapi import FastAPI, Request
import socketio


class Node(FastAPI):
    def __init__(self):
        super().__init__()
        self.sio = socketio.AsyncClient(
            reconnection_delay=0,
            request_timeout=0.5,
            # logger=True, engineio_logger=True
        )
