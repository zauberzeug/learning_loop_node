
from learning_loop_node.node import Node
from fastapi import APIRouter,  Request, HTTPException
from fastapi.responses import PlainTextResponse
import logging
from enum import Enum

router = APIRouter()


class OperationMode(str, Enum):
    Startup = 'startup'  # used until model is loaded
    Idle = 'idle'  # will check and perform updates
    Detecting = 'detecting'  # Blocks updates


@router.put("/operation_mode")
async def operation_mode(request: Request):
    '''
    Example Usage
        curl -X PUT -d "check_for_updates" http://localhost/operation_mode
        curl -X PUT -d "detecting" http://localhost/operation_mode
    '''

    content = str(await request.body(), 'utf-8')
    try:
        target_mode = OperationMode(content)
    except ValueError as e:
        raise HTTPException(422, str(e))

    node: Node = request.app

    logging.info(f'current node state : {node.status.state}')
    logging.info(f'target operation mode : {target_mode}')

    await node.set_operation_mode(target_mode)
    return "OK"


@router.get("/operation_mode")
async def operation_mode(request: Request):
    '''
    Example Usage
        curl http://localhost/operation_mode
    '''
    return PlainTextResponse(request.app.operation_mode.value)
