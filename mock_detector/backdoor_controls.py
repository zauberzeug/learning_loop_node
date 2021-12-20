"""These restful endpoints are only to be used for testing purposes and are not part of the 'offical' trainer behavior."""
from learning_loop_node.detector.detector_node import DetectorNode
from fastapi import APIRouter,  Request
import logging
from learning_loop_node.globals import GLOBALS
import shutil
router = APIRouter()


@router.put("/socketio")
async def switch_socketio(request: Request):
    logging.error('############ hier switch_socketio')
    '''
    Example Usage

        curl -X PUT -d "on" http://localhost:8007/socketio
    '''
    state = str(await request.body(), 'utf-8')
    await _switch_socketio(state, request.app)


async def _switch_socketio(state: str, detector_node: DetectorNode):
    if state == 'off':
        logging.debug('turning socketio off')
        await detector_node.sio_client.disconnect()
    if state == 'on':

        logging.debug('turning socketio on')
        await detector_node.connect()


@router.post("/reset")
async def switch_socketio(request: Request):
    shutil.rmtree(GLOBALS.data_folder, ignore_errors=True)
    request.app.reload()
