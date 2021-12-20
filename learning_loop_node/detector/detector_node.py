from learning_loop_node.detector import operation_mode
from learning_loop_node.detector.operation_mode import OperationMode
from learning_loop_node.node import Node
from learning_loop_node.status import State
from learning_loop_node.status import DetectionStatus, State
from learning_loop_node.context import Context
from learning_loop_node.rest import downloads
from fastapi.encoders import jsonable_encoder
from fastapi_utils.tasks import repeat_every
from typing import Union
import shutil
import os
import logging
from learning_loop_node.globals import GLOBALS
from icecream import ic
import json
import subprocess

class DetectorNode(Node):
    current_model_id: Union[str, None] = None
    target_model_id: Union[str, None] = None
    organization: str
    project: str
    operation_mode: OperationMode = OperationMode.Idle

    def __init__(self, name: str, uuid: str = None):
        super().__init__(name, uuid)

        self.organization = os.environ.get('LOOP_ORGANIZATION', None) or os.environ.get('ORGANIZATION', None)
        self.project = os.environ.get('LOOP_PROJECT', None) or os.environ.get('PROJECT', None)
        assert self.organization, 'Detector node needs an organization'
        assert self.project, 'Detector node needs an project'

        self.include_router(operation_mode.router, prefix="")

        @self.on_event("startup")
        @repeat_every(seconds=1, raise_exceptions=False, wait_first=False)
        async def _check_for_update() -> None:
            await self.check_for_update()

        try:
            with open(f'{GLOBALS.data_folder}/model/model.json', 'r') as f:
                content = json.load(f)
                self.current_model_id = content['id']
        except:
            pass

    async def check_for_update(self):
        if self.operation_mode == OperationMode.Check_for_updates:
            logging.info('going to check for new updates')
            await self.send_status()
            if self.target_model_id != self.current_model_id:
                target_model_folder = f'{GLOBALS.data_folder}/models/{self.target_model_id}'
                shutil.rmtree(target_model_folder, ignore_errors=True)
                os.makedirs(target_model_folder)
                await downloads.download_model(target_model_folder, Context(organization=self.organization, project=self.project), self.target_model_id, 'mocked')
                shutil.rmtree(f'{GLOBALS.data_folder}/model', ignore_errors=True)
                shutil.copytree(target_model_folder, f'{GLOBALS.data_folder}/model')

                self.reload()

    async def send_status(self) -> dict:
        status = DetectionStatus(
            id=self.uuid,
            name=self.name,
            state=self.status.state,
            operation_mode=self.operation_mode,
            current_model_id=self.current_model_id,
            target_model_id=self.target_model_id,
            latest_error=self.status.latest_error,
        )

        logging.info(f'sending status {status}')
        response = await self.sio_client.call('update_detector', (self.organization, self.project, jsonable_encoder(status)), timeout=1)
        self.target_model_id = response['payload']['target_model_id']
        return True

    def get_state(self):
        return State.Online

    async def set_operation_mode(self, mode: OperationMode):
        self.operation_mode = mode
        await self.send_status()

    def reload(self):
        subprocess.call(["touch", "/app/restart/restart.py"])
