import asyncio
import shutil
from log_parser import LogParser
from threading import Thread
import backdoor_controls
from fastapi_utils.tasks import repeat_every
import io
from learning_loop_node.node import Node
import learning_loop_node.node_helper as node_helper
import os
from typing import List, Union
import helper
import yolo_helper
import yolo_cfg_helper
from uuid import uuid4
import os
from glob import glob
import subprocess
from icecream import ic
import psutil
from status import Status
from uuid import uuid4
import traceback
import helper
from learning_loop_node.trainer.trainer import Trainer
from learning_loop_node.trainer.model import Model
from fastapi.encoders import jsonable_encoder


async def check_state(training_id: str, node: Trainer) -> None:
    model = _parse_latest_iteration(training_id, node)
    if model:
        last_published_iteration = node.training.last_published_iteration
        if not last_published_iteration or model['iteration'] > last_published_iteration:
            model_id = str(uuid4())
            training_path = helper.get_training_path_by_id(training_id)
            weightfile_name = model['weightfile']
            if not weightfile_name:
                return
            weightfile_path = f'{training_path}/{weightfile_name}'
            shutil.move(weightfile_path, f'{training_path}/{model_id}.weights')

            new_model = Model(
                id=model_id,
                hyperparameters=node.training.last_known_model.hyperparameters,
                confusion_matrix=node.training.last_known_model.confusion_matrix,
                parent_id=node.training.last_known_model.id,
                train_image_count=node.training.last_known_model.train_image_count,
                test_image_count=node.training.last_known_model.test_image_count,
                trainer_id=node.status.id,
            )
            await node.sio.call('update_model', (node.training.organization, node.training.project, jsonable_encoder(new_model)))
            node.training.last_produced_model = new_model
            node.training.last_published_iteration = model['iteration']


def _parse_latest_iteration(training_id: str, node: Trainer) -> Union[dict, None]:
    training_path = helper.get_training_path_by_id(training_id)
    log_file_path = f'{training_path}/last_training.log'

    with open(log_file_path, 'r') as f:
        log = f.read()

    iteration_log = LogParser.extract_iteration_log(log)
    if not iteration_log:
        return None

    parser = LogParser(iteration_log)
    iteration = parser.parse_iteration()

    confusion_matrices = {}
    for parsed_class in parser.parse_classes():
        name = parsed_class['name']
        id = _get_id_of_category_from_name(name, node.training.data.box_categories)
        del parsed_class['id']
        del parsed_class['name']
        confusion_matrices[id] = parsed_class

    weightfile = parser.parse_weightfile()
    return {'iteration': iteration, 'confusion_matrix': confusion_matrices, 'weightfile': weightfile}


def _get_id_of_category_from_name(name: str, box_categories: List[dict]) -> str:
    category_id = [category['id'] for category in box_categories if category['name'] == name]
    return category_id[0]
