from typing import Optional
from dataclasses import dataclass
from pydantic import BaseModel


class Model(BaseModel):
    id: str
    hyperparameters: Optional[dict]
    confusion_matrix: Optional[dict]
    parent_id: Optional[str]
    train_image_count: Optional[int]
    test_image_count: Optional[int]
    trainer_id: Optional[str]
