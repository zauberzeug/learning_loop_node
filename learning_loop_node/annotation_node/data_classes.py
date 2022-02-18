from dataclasses import dataclass
from typing import List, Optional
from pydantic import BaseModel
from enum import Enum
from learning_loop_node.context import Context
from learning_loop_node.data_classes.category import Category


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
    LeftMouseDown = 'left_mouse_down',
    RightMouseDown = 'right_mouse_down',
    MouseMove = 'mouse_move',
    LeftMouseUp = 'left_mouse_up',
    RightMouseUp = 'right_mouse_up',
    KeyUp = 'key_up',
    KeyDown = 'key_down',


class AnnotationData(BaseModel):
    coordinate: Point
    event_type: EventType
    context: Context
    image_uuid: str
    category: Category
    is_shift_key_pressed: Optional[bool]
    key_up: Optional[str] = None
    key_down: Optional[str] = None
    # keyboard_modifiers: Optional[List[str]]
    # new_annotation_uuid: Optional[str]
    # edit_annotation_uuid: Optional[str]


class UserInput(BaseModel):
    frontend_id: str
    data: AnnotationData


class ToolOutput(BaseModel):
    svg: str
    annotation: Optional[SegmentationAnnotation]
