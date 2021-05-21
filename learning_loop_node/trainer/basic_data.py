from pydantic.main import BaseModel
from typing import List


class BasicData(BaseModel):
    image_ids: List[str]
    box_categories: List[dict]
