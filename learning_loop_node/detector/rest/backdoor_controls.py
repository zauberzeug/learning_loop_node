"""These restful endpoints are only to be used for testing purposes and are not part of the 'offical' trainer behavior."""
import logging
import os
import shutil
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request

from ...globals import GLOBALS

if TYPE_CHECKING:
    from ..detector_node import DetectorNode

router = APIRouter()


@router.post("/reset")
async def _reset(request: Request):
    '''
    Soft-Reset the detector node.

    Example Usage

        curl -X POST http://hosturl/reset
    '''
    logging.info('BC: reset')
    detector_node: 'DetectorNode' = request.app

    try:
        shutil.rmtree(GLOBALS.data_folder, ignore_errors=True)
        os.makedirs(GLOBALS.data_folder, exist_ok=True)

        await detector_node.soft_reload()
    except Exception:
        logging.exception('BC: could not reset:')
        return False
    return True
