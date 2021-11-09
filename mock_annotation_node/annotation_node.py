from learning_loop_node.node import Node
import logging
import os
from fastapi.encoders import jsonable_encoder
from learning_loop_node.status import State, TrainingStatus


class AnnotationNode(Node):

    def __init__(self):
        super().__init__('Annotation Node ' + os.uname()[1], '00000000-1111-2222-3333-444444444444')

        @self.sio_client.on('save')
        def on_user_input(organization, project, model):
            return "some_svg_response_from_node"

    async def send_status(self):
        status = TrainingStatus(
            id=self.uuid,
            name=self.name,
            state=State.Idle,
        )

        logging.info(f'sending status {status}')
        result = await self.sio_client.call('update_trainer', jsonable_encoder(status), timeout=1)
        if result != True:
            raise Exception(result)
