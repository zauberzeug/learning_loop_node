from learning_loop_node.trainer.training import Training
import os


def file_path(training: Training) -> str:
    return f'{training.training_folder}/detection_uploading_progress.txt'


def save(training: Training, count: int) -> None:
    with open(file_path(training), 'w') as f:
        f.write(str(count))


def load(training: Training) -> int:
    if not exists(training):
        return 0
    with open(file_path(training), 'r') as f:
        return int(f.read())


def delete(training) -> None:
    path = file_path(training)
    if exists(training):
        os.remove(file_path(training))


def exists(training) -> None:
    return os.path.exists(file_path(training))
