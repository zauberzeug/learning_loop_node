import random
from typing import Dict, Optional

from learning_loop_node.data_classes import TrainingStateData
from learning_loop_node.trainer.trainer_logic import TrainerLogic


def increment_time(trainer: TrainerLogic, latest_known_confusion_matrix: Dict) -> Optional[TrainingStateData]:
    if not trainer._training:  # pylint: disable=protected-access
        return None

    confusion_matrix = {}
    for category in trainer.training.categories:
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

    new_model = TrainingStateData(
        confusion_matrix=confusion_matrix,
    )

    return new_model
