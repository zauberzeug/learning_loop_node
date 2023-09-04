"""These restful endpoints are only to be used for testing purposes and are not part of the 'offical' trainer behavior."""

import logging

from fastapi import APIRouter, Request

from learning_loop_node.annotation.annotator_node import AnnotatorNode
from learning_loop_node.status import NodeState

router = APIRouter()


@router.put("/socketio")
async def switch_socketio(request: Request):
    state = str(await request.body(), 'utf-8')
    await _switch_socketio(state, request.app)


async def _switch_socketio(state: str, annotator_node: AnnotatorNode):
    if state == 'off':
        if annotator_node.status.state != NodeState.Offline:
            logging.debug('turning socketio off')
            assert annotator_node._sio_client is not None
            await annotator_node._sio_client.disconnect()
    if state == 'on':
        if annotator_node.status.state == NodeState.Offline:
            logging.debug('turning socketio on')
            await annotator_node.connect()
