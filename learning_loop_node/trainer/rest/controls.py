
from learning_loop_node.context import Context
from learning_loop_node.node import Node
from learning_loop_node.loop import loop
from fastapi import APIRouter,  Request, HTTPException
from fastapi.responses import PlainTextResponse
import logging

from learning_loop_node.trainer.trainer import Trainer

router = APIRouter()


@router.post("/controls/detect/{organization}/{project}/{version}")
async def operation_mode(organization: str, project: str, version: str, request: Request):
    '''
    Example Usage
        curl -X POST localhost/controls/detect/<organization>/<project>/<model_version>
    '''
    path = f'/api/{organization}/projects/{project}/models'
    async with loop.get(path) as response:
        if response.status != 200:
            raise HTTPException(404, 'could not load latest model')
        models = (await response.json())['models']
        model_id = next(m for m in models if m['version'] == version)['id']
        logging.info(model_id)
    trainer: Trainer = request.app.trainer
    await trainer.do_detections(Context(organization=organization, project=project), model_id)
    return "OK"
