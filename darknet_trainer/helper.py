from learning_loop_node.training_data import TrainingData
from typing import List
from glob import glob


def get_box_category_ids(training_data: TrainingData) -> List[str]:
    return [c['id']for c in training_data.box_categories]


def get_box_category_names(training_data: TrainingData) -> List[str]:
    return [c['name']for c in training_data.box_categories]


def get_training_path_by_id(trainings_id: str) -> str:
    trainings = [training_path for training_path in glob(
        f'/data/**/trainings/{trainings_id}', recursive=True)]
    return trainings[0]
