"""These restful endpoints are only to be used for testing purposes and are not part of the 'offical' trainer behavior."""

from fastapi import APIRouter, Body, Request, Depends, HTTPException
from typing import List, Optional
from pydantic import BaseModel
import asyncio
from learning_loop_node.status import Status, State
import results
import os
import glob
import shutil

router = APIRouter()


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


@router.put("/status")
async def set_status(new_status: Status, request: Request):
    print('new status is', new_status, flush=True)
    await request.app.update_status(new_status)


@router.post("/steps")
async def add_steps(request: Request):
    if request.app.status.state != State.Running:
        raise HTTPException(status_code=409, detail="trainer is not running")

    steps = int(str(await request.body(), 'utf-8'))
    print(f'moving {steps} forward', flush=True)
    for i in range(0, steps):
        await results.increment_time(request.app)


@router.get("/echo")
def echo():
    return {'msg': 'hello from trainer'}


@router.delete("/images")
def delete_images():
    shutil.rmtree('../data/zauberzeug/pytest/images', ignore_errors=True)
    return get_images()


@router.get("/images")
def get_images():
    files = glob.glob('../data/zauberzeug/pytest/images/*')
    return {'images': files}
