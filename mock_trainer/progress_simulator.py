import random

from learning_loop_node.trainer.model import BasicModel
from learning_loop_node.trainer.trainer import Trainer


def increment_time(trainer: Trainer, latest_known_confusion_matrix: dict) -> BasicModel:
    if not trainer._training or not trainer._training.data:
        return None

    confusion_matrix = {}
    for category in trainer._training.data.categories:
        try:
            minimum = latest_known_confusion_matrix[category.id]['tp']
        except:
            minimum = 0
        maximum = minimum + 1
        confusion_matrix[category.id] = {
            'tp': random.randint(minimum, maximum),
            'fp': max(random.randint(10-maximum, 10-minimum), 2),
            'fn': max(random.randint(10-maximum, 10-minimum), 2),
        }

    new_model = BasicModel(
        confusion_matrix=confusion_matrix,
    )

    return new_model
