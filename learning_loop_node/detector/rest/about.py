
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from fastapi import APIRouter, Request

from ...data_classes import ModelInformation

if TYPE_CHECKING:
    from ..detector_node import DetectorNode
KWONLY_SLOTS = {'kw_only': True, 'slots': True} if sys.version_info >= (3, 10) else {}

router = APIRouter()


@dataclass(**KWONLY_SLOTS)
class AboutResponse:
    operation_mode: str = field(metadata={"description": "The operation mode of the detector node"})
    state: Optional[str] = field(metadata={
        "description": "The state of the detector node",
        "example": "idle, online, detecting"})
    model_info: Optional[ModelInformation] = field(metadata={
        "description": "Information about the model of the detector node"})
    target_model: Optional[str] = field(metadata={"description": "The target model of the detector node"})
    version_control: str = field(metadata={
        "description": "The version control mode of the detector node",
        "example": "follow_loop, specific_version, pause"})


@router.get("/about", response_model=AboutResponse)
async def get_about(request: Request):
    '''
    Get information about the detector node.

    Example Usage

        curl http://hosturl/about
    '''
    app: 'DetectorNode' = request.app

    response = AboutResponse(
        operation_mode=app.operation_mode.value,
        state=app.status.state,
        model_info=app.detector_logic._model_info,  # pylint: disable=protected-access
        target_model=app.target_model.version if app.target_model is not None else None,
        version_control=app.version_control.value
    )

    return response
