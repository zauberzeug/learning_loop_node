
import sys
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request

from ...data_classes import AboutResponse

if TYPE_CHECKING:
    from ..detector_node import DetectorNode
KWONLY_SLOTS = {'kw_only': True, 'slots': True} if sys.version_info >= (3, 10) else {}

router = APIRouter()


@router.get("/about", response_model=AboutResponse)
async def get_about(request: Request):
    '''
    Get information about the detector node.

    Example Usage

        curl http://hosturl/about
    '''
    app: 'DetectorNode' = request.app
    return app.get_about_response()
