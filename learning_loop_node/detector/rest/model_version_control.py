
import os
from enum import Enum
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request

from ...data_classes import ModelInformation
from ...globals import GLOBALS

if TYPE_CHECKING:
    from ..detector_node import DetectorNode

router = APIRouter()


class VersionMode(str, Enum):
    FollowLoop = 'follow_loop'  # will follow the loop
    SpecificVersion = 'specific_version'  # will follow the specific version
    Pause = 'pause'  # will pause the updates


@router.get("/model_version")
async def get_version(request: Request):
    '''
    Example Usage
        curl http://localhost/model_version
    '''
    # pylint: disable=protected-access

    app: 'DetectorNode' = request.app

    current_version = app.detector_logic._model_info.version if app.detector_logic._model_info is not None else 'None'
    target_version = app.target_model.version if app.target_model is not None else 'None'
    loop_version = app.loop_deployment_target.version if app.loop_deployment_target is not None else 'None'

    local_versions: list[str] = []

    local_models = os.listdir(os.path.join(GLOBALS.data_folder, 'models'))
    for model in local_models:
        if model.replace('.', '').isdigit():
            local_versions.append(model)

    return {
        'current_version': current_version,
        'target_version': target_version,
        'loop_version': loop_version,
        'local_versions': local_versions,
        'version_control': app.version_control.value,
    }


@router.put("/model_version")
async def put_version(request: Request):
    '''
    Example Usage
        curl -X PUT -d "follow_loop" http://localhost/model_version
        curl -X PUT -d "pause" http://localhost/model_version
        curl -X PUT -d "13.6" http://localhost/model_version
    '''
    app: 'DetectorNode' = request.app
    content = str(await request.body(), 'utf-8')

    if content == 'follow_loop':
        app.version_control = VersionMode.FollowLoop
    elif content == 'pause':
        app.version_control = VersionMode.Pause
    else:
        app.version_control = VersionMode.SpecificVersion
        if not content or not content.replace('.', '').isdigit():
            raise HTTPException(400, 'Invalid version number')
        target_version = content

        if app.target_model is not None and app.target_model.version == target_version:
            return "OK"

        # Fetch the model uuid by version from the loop
        uri = f'/{app.organization}/projects/{app.project}/models'
        response = await app.loop_communicator.get(uri)
        if response.status_code != 200:
            app.version_control = VersionMode.Pause
            raise HTTPException(500, 'Failed to load models from learning loop')

        models = response.json()['models']
        models_with_target_version = [m for m in models if m['version'] == target_version]
        if len(models_with_target_version) == 0:
            app.version_control = VersionMode.Pause
            raise HTTPException(400, f'No Model with version {target_version}')
        if len(models_with_target_version) > 1:
            app.version_control = VersionMode.Pause
            raise HTTPException(500, f'Multiple models with version {target_version}')

        model_id = models_with_target_version[0]['id']
        model_host = models_with_target_version[0].get('host', 'unknown')

        app.target_model = ModelInformation(organization=app.organization, project=app.project,
                                            host=model_host, categories=[],
                                            id=model_id,
                                            version=target_version)

    return "OK"
