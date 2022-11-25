from . import detections
from . import detections_upload_progress
from . import model_upload_progress
import json
from fastapi.encoders import jsonable_encoder
from learning_loop_node.trainer.training import Training
import os
from learning_loop_node.globals import GLOBALS

node_uuid = None


def init(uuid: str) -> None:
    global node_uuid
    node_uuid = uuid


def file_path() -> str:
    global node_uuid
    if not node_uuid:
        raise Exception('node_uuid not set. You have to call init(uuid: str) first')
    return f'{GLOBALS.data_folder}/last_training__{node_uuid}.json'


def save(training: Training):
    with open(file_path(), 'w') as f:
        json.dump(jsonable_encoder(training), f)


def load() -> Training:
    with open(file_path(), 'r') as f:
        return Training(**json.load(f))


def delete() -> None:
    if exists():
        os.remove(file_path())


def exists() -> bool:
    return os.path.exists(file_path())
