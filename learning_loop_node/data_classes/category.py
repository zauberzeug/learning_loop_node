from typing import Optional
from pydantic.main import BaseModel
from enum import Enum


class CategoryType(str, Enum):
    Box = 'box'
    Point = 'point'
    Segmentation = 'segmentation'


class Category(BaseModel):
    id: str
    name: str
    description: Optional[str]
    hotkey: Optional[str]
    color: Optional[str]
    type: CategoryType = CategoryType.Box


def create_category(id: str, name: str, type: CategoryType):
    return Category(id=id, name=name, description='', hotkey='', color='', type=type)
