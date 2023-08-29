import os

from learning_loop_node.trainer.training import Training


def file_path(training: Training) -> str:
    return f'{training.training_folder}/detection_uploading_json_index.txt'


def save(training: Training, index: int) -> None:
    with open(file_path(training), 'w') as f:
        f.write(str(index))


def load(training: Training) -> int:
    if not exists(training):
        return 0
    with open(file_path(training), 'r') as f:
        return int(f.read())


def delete(training) -> None:
    if exists(training):
        os.remove(file_path(training))


def exists(training) -> None:
    return os.path.exists(file_path(training))
