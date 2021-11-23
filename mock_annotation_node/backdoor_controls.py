"""These restful endpoints are only to be used for testing purposes and are not part of the 'offical' trainer behavior."""

from learning_loop_node.trainer.error_configuration import ErrorConfiguration
from fastapi import APIRouter,  Request,  HTTPException
from learning_loop_node.status import Status, State
import logging
from learning_loop_node.annotation_node.annotation_node import AnnotationNode

router = APIRouter()


@router.put("/socketio")
async def switch_socketio(request: Request):
    state = str(await request.body(), 'utf-8')
    await _switch_socketio(state, request.app)


async def _switch_socketio(state: str, annotation_node: AnnotationNode):
    if state == 'off':
        if annotation_node.status.state != State.Offline:
            logging.debug('turning socketio off')
            await annotation_node.sio_client.disconnect()
    if state == 'on':
        if annotation_node.status.state == State.Offline:
            logging.debug('turning socketio on')
            await annotation_node.connect()
