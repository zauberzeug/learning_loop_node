from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel


class BasicModel(BaseModel):
    confusion_matrix: Optional[dict]
    meta_information: Optional[dict]


class Model(BaseModel):
    id: str
    confusion_matrix: Optional[dict]
    parent_id: Optional[str]
    train_image_count: Optional[int]
    test_image_count: Optional[int]
    trainer_id: Optional[str]
    hyperparameters: Optional[str]


@dataclass
class PretrainedModel():
    name: str
    label: str
    description: str
