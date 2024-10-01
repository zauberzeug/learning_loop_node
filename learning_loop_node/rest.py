import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request

if TYPE_CHECKING:
    from .node import Node


router = APIRouter()


@router.put("/debug_logging")
async def _debug_logging(request: Request):
    '''
    Example Usage

        curl -X PUT -d "on" http://localhost:8007/debug_logging
    '''
    state = str(await request.body(), 'utf-8')
    node: 'Node' = request.app

    if state == 'off':
        logging.info('Node: turning debug logging off')
        node.log.setLevel('INFO')
    if state == 'on':
        logging.info('Node: turning debug logging on')
        node.log.setLevel('DEBUG')
