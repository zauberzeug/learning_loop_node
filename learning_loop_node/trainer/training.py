from enum import Enum
from learning_loop_node.trainer.training_data import TrainingData
from pydantic import BaseModel
from typing import Optional
from learning_loop_node.context import Context
import json


class Training(BaseModel):
    base_model_id: Optional[str]
    id: str
    context: Context

    project_folder: str
    images_folder: str

    training_folder: Optional[str]

    data: Optional[TrainingData]

    training_number: Optional[int]

    training_state: Optional[str]
    training_sub_state: Optional[str]


def save(training: Training):
    with open('last_training.json', 'w') as f:
        json.dump(training.dict(), f)


def load() -> Training:
    with open('last_training.json', 'r') as f:
        return Training(**json.load(f))


class TrainingOut(BaseModel):
    confusion_matrix: Optional[dict]
    train_image_count: Optional[int]
    test_image_count: Optional[int]
    trainer_id: Optional[str]
    hyperparameters: Optional[dict]


class State(str, Enum):
    Preparing = 'preparing'
