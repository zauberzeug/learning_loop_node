from ast import List
from learning_loop_node.trainer.training import Training
import os
import json
from fastapi.encoders import jsonable_encoder
from pathlib import Path


def file_path(training: Training, index: int = 0) -> str:
    return f'{training.training_folder}/detections_{index}.json'


def get_file_names(training: Training) -> List:
    files = [f for f in Path(training.training_folder).iterdir()
             if f.is_file() and f.name.startswith('detections_')]
    if not files:
        return []
    return files


def save(training: Training, detections: List, index: int = 0) -> None:
    with open(file_path(training, index), 'w') as f:
        json.dump(jsonable_encoder(detections), f)


def load(training: Training, index: int = 0) -> List:
    with open(file_path(training, index), 'r') as f:
        return json.load(f)


def delete(training: Training) -> None:
    number_of_files = len(get_file_names(training))
    for i in range(number_of_files):
        if exists(training):
            os.remove(file_path(training, i))


def exists(training: Training) -> bool:
    return os.path.exists(file_path(training))
