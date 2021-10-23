from pydantic.main import BaseModel
from .context import Context
from typing import List, Optional
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


class ModelInformation(BaseModel):
    id: str
    host: str
    organization: str
    project: str
    version: str
    categories: List[Category]
    resolution: Optional[int]

    @property
    def context(self):
        return Context(organization=self.organization, project=self.project)
