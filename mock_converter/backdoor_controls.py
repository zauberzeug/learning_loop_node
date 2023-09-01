"""These restful endpoints are only to be used for testing purposes and are not part of the 'offical' trainer behavior."""

import asyncio

from fastapi import APIRouter, HTTPException, Request

from learning_loop_node.status import State

router = APIRouter()


@router.put("/socketio")
async def put_socketio(request: Request):
    '''
    Example Usage

        curl -X PUT -d "on" http://localhost:8005/socketio
    '''
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


@router.put("/check_state")
async def put_check_state(request: Request):
    value = str(await request.body(), 'utf-8')
    if value == 'off':
        request.app.skip_check_state = True
        for i in range(5):
            if request.app.status.state != State.Idle:
                await asyncio.sleep(0.5)
            else:
                break
        if request.app.status.state != State.Idle:
            raise HTTPException(status_code=409, detail="Could not skip auto checking. State is still not idle")

        print(f'turning automatically check_state {value}', flush=True)
    if value == 'on':
        request.app.skip_check_state = False
        print(f'turning automatically check_state {value}', flush=True)


@router.post("/step")
async def add_steps(request: Request):
    if request.app.status.state == State.Running:
        raise HTTPException(status_code=409, detail="converter is already running")

    await request.app.check_state()
