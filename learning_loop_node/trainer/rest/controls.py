
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse

from learning_loop_node.data_classes.context import Context
from learning_loop_node.loop_communication import glc
from learning_loop_node.node import Node
from learning_loop_node.trainer.trainer import Trainer

router = APIRouter()


@router.post("/controls/detect/{organization}/{project}/{version}")
async def operation_mode(organization: str, project: str, version: str, request: Request):
    '''
    Example Usage
        curl -X POST localhost/controls/detect/<organization>/<project>/<model_version>
    '''
    path = f'/{organization}/projects/{project}/models'
    response = await glc.get(path)
    if response.status_code != 200:
        raise HTTPException(404, 'could not load latest model')
    models = response.json()['models']
    model_id = next(m for m in models if m['version'] == version)['id']
    logging.info(model_id)
    trainer: Trainer = request.app.trainer
    await trainer.do_detections(Context(organization=organization, project=project), model_id)
    return "OK"
