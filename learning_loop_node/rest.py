import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request

if TYPE_CHECKING:
    from .node import Node


router = APIRouter()
logger = logging.getLogger('Node.rest')


@router.put("/debug_logging")
async def _debug_logging(request: Request):
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
async def _socketio(request: Request):
    '''
    Example Usage

        curl -X PUT -d "on" http://localhost:8007/socketio
    '''
    state = str(await request.body(), 'utf-8')
    node: 'Node' = request.app

    if state == 'off':
        logging.info('BC: turning socketio off')
        await node.sio_client.disconnect()
    if state == 'on':
        logging.info('BC: turning socketio on')
        await node.reset_loop_connection()
        # await detector_node.reset_sio_connection()
