from ast import List
from learning_loop_node.trainer.training import Training
import os
import json


def file_path(training: Training) -> str:
    return f'{training.training_folder}/detections.json'


def save(training: Training, detections: List) -> None:
    with open(file_path(training), 'w') as f:
        json.dump(detections, f)


def load(training: Training,) -> List:
    with open(file_path(training), 'r') as f:
        return json.load(f)


def delete(training: Training) -> None:
    if exists(training):
        os.remove(file_path(training))


def exists(training: Training) -> bool:
    return os.path.exists(file_path(training))
