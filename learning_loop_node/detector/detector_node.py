from typing import Union
from learning_loop_node.node import Node
from learning_loop_node.status import State
import os

from learning_loop_node.status import DetectionStatus, State
from learning_loop_node.context import Context
import logging
from fastapi.encoders import jsonable_encoder


class DetectorNode(Node):
    current_model_id: Union[str, None] = None
    target_model_id: Union[str, None] = None
    organization: str
    project: str

    def __init__(self, name: str, uuid: str = None):
        super().__init__(name, uuid)

        self.organization = os.environ.get('LOOP_ORGANIZATION', None) or os.environ.get('ORGANIZATION', None)
        self.project = os.environ.get('LOOP_PROJECT', None) or os.environ.get('PROJECT', None)
        assert self.organization, 'Detector node needs an organization'
        assert self.project, 'Detector node needs an project'

    async def send_status(self):
        status = DetectionStatus(
            id=self.uuid,
            name=self.name,
            state=self.status.state,
            current_model_id=self.current_model_id,
            target_model_id=self.target_model_id,
            latest_error=self.status.latest_error,
        )
        logging.info(f'sending status {status}')
        result = await self.sio_client.call('update_detector', (self.organization, self.project, jsonable_encoder(status)), timeout=1)
        if not result == True:
            raise Exception(result)
        logging.info('status send')

    def get_state(self):
        return State.Online
