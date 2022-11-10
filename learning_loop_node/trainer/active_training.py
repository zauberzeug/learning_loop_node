from enum import Enum
from learning_loop_node.trainer.training_data import TrainingData
from pydantic import BaseModel
from typing import List, Optional
from learning_loop_node.context import Context
import json
from fastapi.encoders import jsonable_encoder
from learning_loop_node.trainer.training import Training
import os
from learning_loop_node.globals import GLOBALS


def training_file_path() -> str:
    return f'{GLOBALS.data_folder}/last_training.json'


def detection_file_path(training: Training) -> str:
    return f'{training.training_folder}/detections.json'


def save(training: Training):
    with open(training_file_path(), 'w') as f:
        json.dump(jsonable_encoder(training), f)


def load() -> Training:
    with open(training_file_path(), 'r') as f:
        return Training(**json.load(f))


def delete() -> None:
    if exists():
        os.remove(training_file_path())


def exists() -> bool:
    return os.path.exists(training_file_path())


def save_detections(training: Training, detections: List) -> None:
    with open(detection_file_path(training), 'w') as f:
        json.dump(detections, f)


def load_detections(training: Training,) -> List:
    with open(detection_file_path(training), 'r') as f:
        return json.load(f)


def detections_exist(training: Training) -> bool:
    return os.path.exists(detection_file_path(training))


def delete_detections(training: Training) -> None:
    if detections_exist(training):
        os.remove(detection_file_path(training))
