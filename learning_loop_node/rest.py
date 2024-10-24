import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request

if TYPE_CHECKING:
    from .node import Node


router = APIRouter()
logger = logging.getLogger('Node.rest')


@router.put("/debug_logging")
async def _debug_logging(request: Request) -> str:
    '''
    Example Usage

        curl -X PUT -d "on" http://localhost:8007/debug_logging
    '''
    state = str(await request.body(), 'utf-8')
    node: 'Node' = request.app

    if state == 'off':
        logger.info('turning debug logging off')
        node.log.setLevel('INFO')
        return 'off'
    if state == 'on':
        logger.info('turning debug logging on')
        node.log.setLevel('DEBUG')
        return 'on'
    raise HTTPException(status_code=400, detail='Invalid state')


@router.put("/socketio")
async def _socketio(request: Request) -> str:
    '''
    Enable or disable the socketio connection to the learning loop.
    Not intended to be used outside of testing.

    Example Usage

        curl -X PUT -d "on" http://hosturl/socketio
    '''
    state = str(await request.body(), 'utf-8')
    node: 'Node' = request.app

    if state == 'off':
        await node.sio_client.disconnect()
        node.set_skip_repeat_loop(True)  # Prevent auto-reconnection
        return 'off'
    if state == 'on':
        node.set_skip_repeat_loop(False)  # Allow auto-reconnection (1 sec delay)
        return 'on'
    raise HTTPException(status_code=400, detail='Invalid state')
