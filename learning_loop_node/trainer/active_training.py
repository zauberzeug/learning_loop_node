from enum import Enum
from learning_loop_node.trainer.training_data import TrainingData
from pydantic import BaseModel
from typing import Optional
from learning_loop_node.context import Context
import json
from fastapi.encoders import jsonable_encoder
from learning_loop_node.trainer.training import Training
import os
from learning_loop_node.globals import GLOBALS

file_path = f'{GLOBALS.data_folder}/last_training.json'


def save(training: Training):
    with open(file_path, 'w') as f:
        json.dump(jsonable_encoder(training), f)


def load() -> Training:
    with open(file_path, 'r') as f:
        return Training(**json.load(f))


def delete() -> None:
    os.remove(file_path)


def exists() -> bool:
    return os.path.exists(file_path)
