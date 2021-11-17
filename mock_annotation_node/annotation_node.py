from learning_loop_node.node import Node
import logging
import os
from fastapi.encoders import jsonable_encoder
from learning_loop_node.status import State, AnnotationNodeStatus
from icecream import ic
from pydantic import BaseModel
import logging


class SvgBox(BaseModel):
    x: int
    y: int
    x2: int
    y2: int

    def __str__(self):
        width = abs(self.x2-self.x)
        height = abs(self.y2-self.y)
        return f'<rect x="{min(self.x, self.x2)}" y="{min(self.y, self.y2)}" width="{width or 1}" height="{height or 1}" fill="blue">'


class AnnotationNode(Node):
    def __init__(self):
        super().__init__('Annotation Node ' + os.uname()[1], '00000000-1111-2222-3333-444444444444')
        self.box: SvgBox = None

        @self.sio_client.on('handle_user_input')
        def on_handle_user_input(organization, project, user_input):
            ic(f'received user input: {jsonable_encoder(user_input)}')
            data = user_input['data']
            event_type = data['event_type']
            try:
                if event_type == 'mouse_down':
                    coordinates = data['coordinates']
                    self.box = SvgBox(x=coordinates['x'], y=coordinates['y'], x2=coordinates['x'], y2=coordinates['y'])
                    return str(self.box)
                if event_type == 'mouse_move':
                    coordinates = data['coordinates']
                    self.box.x2 = coordinates['x']
                    self.box.y2 = coordinates['y']
                    return str(self.box)
                if event_type == 'mouse_up':
                    self.box = None
                    return ""
            except:
                return ""

            return "some_svg_response_from_node"

    async def send_status(self):
        status = AnnotationNodeStatus(
            id=self.uuid,
            name=self.name,
            state=State.Online,
            capabilities=['segmentation']
        )

        logging.info(f'sending status {status}')
        result = await self.sio_client.call('update_annotation_node', jsonable_encoder(status), timeout=2)
        if result != True:
            raise Exception(result)
