
from enum import Enum


class CategoryType(str, Enum):
    Box = 'box'
    Point = 'point'
    Segmentation = 'segmentation'
    Classification = 'classification'
