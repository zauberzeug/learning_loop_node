from typing import List
from learning_loop_node.trainer.training import Training
import os


def file_path(training: Training) -> str:
    return f'{training.training_folder}/model_uploading_progress.txt'


def save(training: Training, formats: List[str]) -> None:
    with open(file_path(training), 'w') as f:
        f.write(','.join(formats))


def load(training: Training) -> List[str]:
    if not exists(training):
        return []
    with open(file_path(training), 'r') as f:
        return f.read().split(',')


def delete(training) -> None:
    path = file_path(training)
    if exists(training):
        os.remove(file_path(training))


def exists(training) -> None:
    return os.path.exists(file_path(training))
