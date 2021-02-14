''' These restful endpoints are only to be used for testing purposes and are not part of the 'offical' trainer behavior.'''

from fastapi import APIRouter, Body, Request, Depends, HTTPException
from typing import List, Optional
from pydantic import BaseModel
import asyncio
import status


router = APIRouter()


@router.put("/socketio")
async def switch_socketio(request: Request):
    state = str(await request.body(), 'utf-8')
    print(status.status, flush=True)
    if state == 'off':
        if status.status.state != status.State.Offline:
            print('turning socketio off', flush=True)
            asyncio.create_task(request.app.sio.disconnect())
    if state == 'on':
        if status.status.state == status.State.Offline:
            print('turning socketio on', flush=True)
            asyncio.create_task(request.app.connect())


@router.put("/status")
async def set_status(new_status: status.Status, request: Request):
    print('new status is', new_status, flush=True)
    await status.update_status(request.app.sio, new_status)


@router.post("/steps")
async def add_steps(request: Request):
    if status.status.state != status.State.Running:
        raise HTTPException(status_code=409, detail="trainer is not running")

    steps = int(str(await request.body(), 'utf-8'))
    print(f'moving {steps} forward', flush=True)
    for i in range(0, steps):
        await status.increment_time(request.app.sio)
