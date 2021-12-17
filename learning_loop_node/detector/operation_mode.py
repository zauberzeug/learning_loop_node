
from learning_loop_node.node import Node
from fastapi import APIRouter,  Request
import logging
from enum import Enum
router = APIRouter()


class OperationMode(str, Enum):
    Detecting = 'detecting'
    Check_for_updates = 'check_for_updates'

    Idle = 'Idle'


@router.put("/operation_mode")
async def operation_mode(request: Request):
    '''
    Example Usage

        curl -X PUT -d "detecting" http://mock_detector/operation_mode
    '''

    target_mode = str(await request.body(), 'utf-8')
    node: Node = request.app

    logging.info(f'current node state : {node.status.state}')
    logging.info(f'target node state : {target_mode}')
    await node.set_operation_mode(target_mode)
