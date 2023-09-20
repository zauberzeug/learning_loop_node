"""These restful endpoints are only to be used for testing purposes and are not part of the 'offical' trainer behavior."""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request

from learning_loop_node.data_classes import NodeState

router = APIRouter()


@router.put("/socketio")
async def put_socketio(request: Request):
    '''
    Example Usage

        curl -X PUT -d "on" http://localhost:8005/socketio
    '''
    state = str(await request.body(), 'utf-8')
    if state == 'off':
        if request.app.status.state != NodeState.Offline:
            logging.info('turning socketio off')
            asyncio.create_task(request.app.sio.disconnect())
    if state == 'on':
        if request.app.status.state == NodeState.Offline:
            logging.info('turning socketio on')
            asyncio.create_task(request.app.connect())


@router.put("/check_state")
async def put_check_state(request: Request):
    value = str(await request.body(), 'utf-8')
    print(f'turning automatically check_state {value}', flush=True)

    if value == 'off':
        request.app.skip_check_state = True
        for _ in range(5):
            if request.app.status.state != NodeState.Idle:
                await asyncio.sleep(0.5)
            else:
                break
        if request.app.status.state != NodeState.Idle:
            raise HTTPException(status_code=409, detail="Could not skip auto checking. State is still not idle")

    if value == 'on':
        request.app.skip_check_state = False


@router.post("/step")
async def add_steps(request: Request):
    if request.app.status.state == NodeState.Running:
        raise HTTPException(status_code=409, detail="converter is already running")

    await request.app.check_state()
