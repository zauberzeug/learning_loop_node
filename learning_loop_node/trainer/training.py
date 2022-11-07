from enum import Enum
from learning_loop_node.trainer.training_data import TrainingData
from pydantic import BaseModel
from typing import Optional
from learning_loop_node.context import Context
import json
from fastapi.encoders import jsonable_encoder


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
        json.dump(jsonable_encoder(training), f)


def load() -> Training:
    with open('last_training.json', 'r') as f:
        return Training(**json.load(f))


def delete() -> None:
    import os
    os.remove('last_training.json')


class TrainingOut(BaseModel):
    confusion_matrix: Optional[dict]
    train_image_count: Optional[int]
    test_image_count: Optional[int]
    trainer_id: Optional[str]
    hyperparameters: Optional[dict]


class State(str, Enum):
    Init = 'init'
    Prepared = 'prepared'
