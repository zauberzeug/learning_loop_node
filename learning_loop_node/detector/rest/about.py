
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request

if TYPE_CHECKING:
    from ..detector_node import DetectorNode

router = APIRouter()


@router.get("/about")
async def get_about(request: Request):
    '''
    Example Usage
        curl http://localhost/about
    '''
    app: 'DetectorNode' = request.app

    return {
        'operation_mode': app.operation_mode.value,
        'state': app.status.state,
        'model_info':  app.detector_logic._model_info,  # pylint: disable=protected-access
        'target_model': app.target_model,  # pylint: disable=protected-access
    }
