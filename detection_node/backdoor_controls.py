"""These restful endpoints are only to be used for testing purposes and are not part of the 'offical' trainer behavior."""

from fastapi import APIRouter, Body, Request, Depends, HTTPException
from typing import List, Optional
from pydantic import BaseModel
import asyncio
from learning_loop_node.status import Status, State

router = APIRouter()

# TODO: Adapt to detection node


@router.put("/socketio")
async def switch_socketio(request: Request):
    state = str(await request.body(), 'utf-8')
    print(request.app.status, flush=True)
    if state == 'off':
        if request.app.status.state != State.Offline:
            print('turning socketio off', flush=True)
            asyncio.create_task(request.app.sio.disconnect())
    if state == 'on':
        if request.app.status.state == State.Offline:
            print('turning socketio on', flush=True)
            asyncio.create_task(request.app.connect())
