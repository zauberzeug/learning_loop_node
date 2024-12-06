import logging
import sys
# pylint: disable=no-name-in-module
from dataclasses import dataclass
from typing import Dict, Optional

from learning_loop_node.annotation.annotator_logic import AnnotatorLogic
from learning_loop_node.data_classes import ToolOutput
from learning_loop_node.enums import AnnotationEventType

# NOTE: This is a mock annotator tool. It is used for testing purposes only.

KWONLY_SLOTS = {'kw_only': True, 'slots': True} if sys.version_info >= (3, 10) else {}


@dataclass(**KWONLY_SLOTS)
class SvgBox():
    x: int
    y: int
    x2: int
    y2: int

    def __str__(self):
        width = abs(self.x2-self.x)
        height = abs(self.y2-self.y)
        return f'<rect x="{min(self.x, self.x2)}" y="{min(self.y, self.y2)}" width="{width or 1}" height="{height or 1}" fill="blue">'


class MockAnnotatorLogic(AnnotatorLogic):
    def __init__(self):  # pylint: disable=super-init-not-called
        super().__init__()
        self.box: Optional[SvgBox] = None

    async def handle_user_input(self, user_input, history: Dict) -> ToolOutput:
        out = ToolOutput(svg="", annotation=None)
        event_type = user_input.data.event_type
        try:
            if user_input.data.event_type == AnnotationEventType.LeftMouseDown:
                coordinate = user_input.data.coordinate
                self.box = SvgBox(x=coordinate.x, y=coordinate.y, x2=coordinate.x, y2=coordinate.y)
                out.svg = str(self.box)
                return out
            if event_type == AnnotationEventType.MouseMove:
                if self.box is None:
                    return out
                coordinate = user_input.data.coordinate
                self.box.x2 = coordinate.x
                self.box.y2 = coordinate.y
                out.svg = str(self.box)
                return out
            if event_type == AnnotationEventType.LeftMouseUp:
                self.box = None
                return out
        except Exception as e:
            logging.error(str(e))
            return out
        out.svg = "some_svg_response_from_node"
        return out

    def create_empty_history(self) -> Dict:
        return {}

    def logout_user(self, sid) -> bool:
        logging.info(f"User {sid} logged out successfully.")
        return True
