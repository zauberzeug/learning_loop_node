import logging
from typing import Optional

# pylint: disable=no-name-in-module
from pydantic import BaseModel

from learning_loop_node.annotation.annotator_model import ToolOutput
from learning_loop_node.annotation.annotator_node import AnnotatorNode
from learning_loop_node.data_classes import EventType

logging.basicConfig(level=logging.INFO)


# NOTE: This is a mock annotator tool. It is used for testing purposes only.
class SvgBox(BaseModel):
    x: int
    y: int
    x2: int
    y2: int

    def __str__(self):
        width = abs(self.x2-self.x)
        height = abs(self.y2-self.y)
        return f'<rect x="{min(self.x, self.x2)}" y="{min(self.y, self.y2)}" width="{width or 1}" height="{height or 1}" fill="blue">'


class MockAnnotatorNode(AnnotatorNode):
    def __init__(self):
        super().__init__()
        self.box: Optional[SvgBox] = None

    async def handle_user_input(self, user_input):
        out = ToolOutput(svg="", annotation=None)
        event_type = user_input.data.event_type
        assert isinstance(self.box, SvgBox)
        try:
            if user_input.data.event_type == EventType.LeftMouseDown:
                coordinate = user_input.data.coordinate
                self.box = SvgBox(x=coordinate.x, y=coordinate.y, x2=coordinate.x, y2=coordinate.y)
                out.svg = str(self.box)
                return out
            if event_type == EventType.MouseMove:
                coordinate = user_input.data.coordinate
                self.box.x2 = coordinate.x
                self.box.y2 = coordinate.y
                out.svg = str(self.box)
                return out
            if event_type == EventType.LeftMouseUp:
                self.box = None
                return out
        except Exception as e:
            logging.error(str(e))
            return out
        out.svg = "some_svg_response_from_node"
        return out

    def create_empty_history(self):
        return {}

    def logout_user(self, sid):
        logging.info(sid)
        return True
