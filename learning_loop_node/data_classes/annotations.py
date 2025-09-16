
import sys
from dataclasses import dataclass, field
from typing import Optional

KWONLY_SLOTS = {'kw_only': True, 'slots': True} if sys.version_info >= (3, 10) else {}


@dataclass(**KWONLY_SLOTS)
class BoxAnnotation():
    """Coordinates according to COCO format. x,y is the top left corner of the box.
    x increases to the right, y increases downwards.
    """
    category_name: str = field(metadata={'description': 'Category name'})
    x: int = field(metadata={'description': 'X coordinate (left to right)'})
    y: int = field(metadata={'description': 'Y coordinate (top to bottom)'})
    width: int = field(metadata={'description': 'Width'})
    height: int = field(metadata={'description': 'Height'})
    category_id: Optional[str] = field(default=None, metadata={'description': 'Category UUID'})

    def __str__(self):
        return f'x:{int(self.x)} y: {int(self.y)}, w: {int(self.width)} h: {int(self.height)} -> {self.category_name}'


@dataclass(**KWONLY_SLOTS)
class PointAnnotation():
    """Coordinates according to COCO format. x,y is the center of the point.
    x increases to the right, y increases downwards."""
    category_name: str = field(metadata={'description': 'Category name'})
    x: float = field(metadata={'description': 'X coordinate (right)'})
    y: float = field(metadata={'description': 'Y coordinate (down)'})
    category_id: Optional[str] = field(default=None, metadata={'description': 'Category UUID'})

    def __str__(self):
        return f'x:{int(self.x)} y: {int(self.y)}, -> {self.category_name}'


@dataclass(**KWONLY_SLOTS)
class ClassificationAnnotation():
    category_name: str = field(metadata={'description': 'Category name'})
    category_id: Optional[str] = field(default=None, metadata={'description': 'Category UUID'})

    def __str__(self):
        return f'-> {self.category_name}'
