import random
from typing import Dict, Optional

from learning_loop_node.data_classes import BasicModel
from learning_loop_node.trainer.trainer_logic import TrainerLogic


def increment_time(trainer: TrainerLogic, latest_known_confusion_matrix: Dict) -> Optional[BasicModel]:
    if not trainer._training or not trainer._training.data:  # pylint: disable=protected-access
        return None

    confusion_matrix = {}
    assert trainer.training.data is not None
    for category in trainer.training.data.categories:
        try:
            minimum = latest_known_confusion_matrix[category.id]['tp']
        except Exception:
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
