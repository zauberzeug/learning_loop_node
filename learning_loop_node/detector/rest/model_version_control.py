
import sys
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request

from ...data_classes import ModelVersionResponse

if TYPE_CHECKING:
    from ..detector_node import DetectorNode
KWONLY_SLOTS = {'kw_only': True, 'slots': True} if sys.version_info >= (3, 10) else {}

router = APIRouter()


@router.get("/model_version", name='Get model version information', response_model=ModelVersionResponse)
async def get_version(request: Request):
    '''
    Get information about the model version control and the current model version.

    Example Usage
        curl http://localhost/model_version
    '''
    # pylint: disable=protected-access
    app: 'DetectorNode' = request.app
    return app.get_model_version_response()


@router.put("/model_version", name='Set model version control mode')
async def put_version(request: Request):
    '''
    Set the model version control mode.

    Example Usage
        curl -X PUT -d "follow_loop" http://hosturl/model_version
        curl -X PUT -d "pause" http://hosturl/model_version
        curl -X PUT -d "13.6" http://hosturl/model_version
    '''
    app: 'DetectorNode' = request.app
    content = str(await request.body(), 'utf-8')
    try:
        await app.set_model_version_mode(content)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc

    return "OK"
