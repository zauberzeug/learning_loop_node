"""These restful endpoints are only to be used for testing purposes and are not part of the 'offical' trainer behavior."""
import logging
import shutil

from fastapi import APIRouter, Request

from learning_loop_node import DetectorNode
from learning_loop_node.globals import GLOBALS

router = APIRouter()


@router.put("/socketio")
async def _socketio(request: Request):
    '''
    Example Usage

        curl -X PUT -d "on" http://localhost:8007/socketio
    '''
    state = str(await request.body(), 'utf-8')
    await _switch_socketio(state, request.app)


async def _switch_socketio(state: str, detector_node: DetectorNode):
    if state == 'off':
        logging.info('BC: turning socketio off')
        await detector_node.sio_client.disconnect()
    if state == 'on':

        logging.info('BC: turning socketio on')
        await detector_node.connect_sio()


@router.post("/reset")
async def _reset(request: Request):
    logging.info('BC: reset')
    try:
        shutil.rmtree(GLOBALS.data_folder, ignore_errors=True)

        # get file dir
        # restart_path = Path(os.path.realpath(__file__)) / 'restart' / 'restart.py'
        # restart_path = Path(os.getcwd()).absolute() / 'app_code' / 'restart' / 'restart.py'
        # restart_path.touch()
        # await request.app.soft_reload()

        assert isinstance(request.app, DetectorNode)
        request.app.reload(reason='------- reset was called from backdoor controls',)
    except Exception as e:
        logging.error(f'BC: could not reset: {e}')
        return False
    return True
