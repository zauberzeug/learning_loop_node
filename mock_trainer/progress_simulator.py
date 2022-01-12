from learning_loop_node.trainer.trainer import Trainer
from learning_loop_node.trainer.model import BasicModel
import random


def increment_time(trainer: Trainer, latest_known_confusion_matrix: dict) -> BasicModel:
    if not trainer.training or not trainer.training.data:
        return None

    confusion_matrix = {}
    for key, value in trainer.training.data.categories.items():
        try:
            minimum = latest_known_confusion_matrix[value]['tp']
        except:
            minimum = 0
        maximum = minimum + 1
        confusion_matrix[value] = {
            'tp': random.randint(minimum, maximum),
            'fp': max(random.randint(10-maximum, 10-minimum), 2),
            'fn': max(random.randint(10-maximum, 10-minimum), 2),
        }

    new_model = BasicModel(
        confusion_matrix=confusion_matrix,
    )

    return new_model
