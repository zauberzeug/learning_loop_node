import sys
from dataclasses import dataclass
from typing import Optional, Union

from ..enums import AnnotationEventType
from .detections import Point, Shape
from .general import Category, Context

KWONLY_SLOTS = {'kw_only': True, 'slots': True} if sys.version_info >= (3, 10) else {}


@dataclass(**KWONLY_SLOTS)
class AnnotationData():
    coordinate: Point
    event_type: Union[AnnotationEventType, str]
    context: Context
    image_uuid: str
    category: Category

    key_up: Optional[str] = None
    key_down: Optional[str] = None
    epsilon: Optional[float] = None
    is_shift_key_pressed: Optional[bool] = None


@dataclass(**KWONLY_SLOTS)
class SegmentationAnnotation():
    id: str
    shape: Shape
    image_id: str
    category_id: str


@dataclass(**KWONLY_SLOTS)
class UserInput():
    frontend_id: str
    data: AnnotationData


@dataclass(**KWONLY_SLOTS)
class ToolOutput():
    svg: str
    annotation: Optional[SegmentationAnnotation] = None
