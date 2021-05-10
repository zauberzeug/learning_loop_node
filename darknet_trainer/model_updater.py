from learning_loop_node.trainer.training import Training
from learning_loop_node.trainer.training_data import TrainingData
import shutil
from log_parser import LogParser
from typing import List, Union
import helper
from uuid import uuid4
from glob import glob
from uuid import uuid4
import helper
from learning_loop_node.trainer.model import BasicModel


def check_state(training_id: str, training_data: TrainingData, last_published_iteration) -> Union[BasicModel, None]:
    model = _parse_latest_iteration(training_id, training_data)
    if model:
        if not last_published_iteration or model['iteration'] > last_published_iteration:
            training_path = helper.get_training_path_by_id(training_id)
            weightfile_name = model['weightfile']
            if not weightfile_name:
                return None
            if not model['confusion_matrix']:
                return None

            weightfile_path = f'{training_path}/{weightfile_name}'
            new_model = BasicModel(
                confusion_matrix=model['confusion_matrix'],
                meta_information={
                    'weightfile_path': weightfile_path,
                    'iteration': model['iteration']
                },
            )
            return new_model


def _parse_latest_iteration(training_id: str, training_data: TrainingData) -> Union[dict, None]:
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
        id = _get_id_of_category_from_name(name, training_data.box_categories)
        del parsed_class['id']
        del parsed_class['name']
        confusion_matrices[id] = parsed_class

    weightfile = parser.parse_weightfile()
    return {'iteration': iteration, 'confusion_matrix': confusion_matrices, 'weightfile': weightfile}


def _get_id_of_category_from_name(name: str, box_categories: List[dict]) -> str:
    category_id = [category['id'] for category in box_categories if category['name'] == name]
    return category_id[0]
