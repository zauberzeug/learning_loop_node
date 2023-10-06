import logging
from enum import Enum
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse

if TYPE_CHECKING:
    from learning_loop_node.detector.detector_node import DetectorNode

router = APIRouter()


class OperationMode(str, Enum):
    Startup = 'startup'  # used until model is loaded
    Idle = 'idle'  # will check and perform updates
    Detecting = 'detecting'  # Blocks updates

# NOTE: This is only ment to be used by a detector node


@router.put("/operation_mode")
async def put_operation_mode(request: Request):
    '''
    Example Usage
        curl -X PUT -d "check_for_updates" http://localhost/operation_mode
        curl -X PUT -d "detecting" http://localhost/operation_mode
    '''

    content = str(await request.body(), 'utf-8')
    try:
        target_mode = OperationMode(content)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    node: DetectorNode = request.app

    logging.info(f'current node state : {node.status.state}')
    logging.info(f'current operation mode : {node.operation_mode.value}')
    logging.info(f'target operation mode : {target_mode}')
    if target_mode == node.operation_mode:
        logging.info('operation mode already set')
        return "OK"

    await node.set_operation_mode(target_mode)
    logging.info(f'operation mode set to : {target_mode}')
    return "OK"


@router.get("/operation_mode")
async def get_operation_mode(request: Request):
    '''
    Example Usage
        curl http://localhost/operation_mode
    '''
    return PlainTextResponse(request.app.operation_mode.value)
