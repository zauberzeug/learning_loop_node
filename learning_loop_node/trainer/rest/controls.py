
import logging

from fastapi import APIRouter, HTTPException, Request

from learning_loop_node.trainer.trainer_logic import TrainerLogic

router = APIRouter()

# pylint: disable=protected-access


@router.post("/controls/detect/{organization}/{project}/{version}")
async def operation_mode(organization: str, project: str, version: str, request: Request):
    '''
    Example Usage
        curl -X POST localhost/controls/detect/<organization>/<project>/<model_version>
    '''
    path = f'/{organization}/projects/{project}/models'
    response = await request.app.loop_communication.get(path)
    if response.status_code != 200:
        raise HTTPException(404, 'could not load latest model')
    models = response.json()['models']
    model_id = next(m for m in models if m['version'] == version)['id']
    logging.info(model_id)
    trainer: TrainerLogic = request.app.trainer
    await trainer._do_detections()
    return "OK"
