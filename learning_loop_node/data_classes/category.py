from typing import List, Optional
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
    point_size: Optional[int]

    @staticmethod
    def from_list(values: List[dict]) -> List['Category']:
        categories: List[Category] = []
        for value in values:
            categories.append(Category.from_dict(value))
        return categories

    @staticmethod
    def from_dict(value: dict) -> 'Category':
        return Category.parse_obj(value)


def create_category(id: str, name: str, type: CategoryType):
    return Category(id=id, name=name, description='', hotkey='', color='', type=type)
