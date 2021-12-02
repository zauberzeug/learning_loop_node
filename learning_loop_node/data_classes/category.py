from pydantic.main import BaseModel
from enum import Enum


class CategoryType(str, Enum):
    Box = 'box'
    Point = 'point'
    Segmentation = 'segmentation'


class Category(BaseModel):
    id: str
    name: str
    description: str
    hotkey: str
    color: str
    type: CategoryType
