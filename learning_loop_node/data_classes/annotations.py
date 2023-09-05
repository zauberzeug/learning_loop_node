from enum import Enum
from typing import Optional

# pylint: disable=no-name-in-module, too-few-public-methods
from pydantic import BaseModel

from learning_loop_node.data_classes.detections import Point, Shape
from learning_loop_node.data_classes.general import Category, Context


class AnnotationEventType(str, Enum):
    LeftMouseDown = 'left_mouse_down'
    RightMouseDown = 'right_mouse_down'
    MouseMove = 'mouse_move'
    LeftMouseUp = 'left_mouse_up'
    RightMouseUp = 'right_mouse_up'
    KeyUp = 'key_up'
    KeyDown = 'key_down'


class AnnotationData(BaseModel):
    coordinate: Point
    event_type: AnnotationEventType
    context: Context
    image_uuid: str
    category: Category

    key_up: Optional[str] = None  # TODO really str???
    key_down: Optional[str] = None
    epsilon: Optional[float] = None
    is_shift_key_pressed: Optional[bool] = None


class SegmentationAnnotation(BaseModel):
    id: str
    shape: Shape
    image_id: str
    category_id: str


class UserInput(BaseModel):
    frontend_id: str
    data: AnnotationData


class ToolOutput(BaseModel):
    svg: str
    annotation: Optional[SegmentationAnnotation] = None
