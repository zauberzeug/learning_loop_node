from typing import List, Optional
from pydantic import BaseModel
from enum import Enum
from learning_loop_node.context import Context
from learning_loop_node.model_information import Category


class Point(BaseModel):
    x: int
    y: int


class Shape(BaseModel):
    points: List[Point]


class SegmentationAnnotation(BaseModel):
    id: str
    shape: Shape
    image_id: str
    category_id: str


class EventType(str, Enum):
    MouseDown = 'mouse_down',
    MouseMove = 'mouse_move',
    MouseUp = 'mouse_up',

    # TODO Introduce Keyboard Event?
    Enter_Pressed = 'enter_pressed'
    ESC_Pressed = 'esc_pressed'


class AnnotationData(BaseModel):
    coordinate: Point
    event_type: EventType
    context: Context
    image_uuid: str
    category: Category
    # keyboard_modifiers: Optional[List[str]]
    # new_annotation_uuid: Optional[str]
    # edit_annotation_uuid: Optional[str]


class UserInput(BaseModel):

    data: AnnotationData


class ToolOutput(BaseModel):
    svg: str
    annotation: Optional[SegmentationAnnotation]
