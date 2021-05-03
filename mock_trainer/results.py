from learning_loop_node.trainer.trainer import Trainer
from learning_loop_node.trainer.model import Model
import uuid
import random


def increment_time(trainer: Trainer):

    if not trainer.training or not trainer.training.data:
        raise Exception("Could not imcrement time.")

    confusion_matrix = {}
    for category in trainer.training.data.box_categories:
        try:
            minimum = trainer.training.last_produced_model['confusion_matrix'][category['id']]['tp']
        except:
            minimum = 0
        maximum = minimum + 1
        confusion_matrix[category['id']] = {
            'tp': random.randint(minimum, maximum),
            'fp': max(random.randint(10-maximum, 10-minimum), 2),
            'fn': max(random.randint(10-maximum, 10-minimum), 2),
        }

    new_model = Model(
        id=str(uuid.uuid4()),
        hyperparameters=trainer.training.last_known_model. hyperparameters,
        confusion_matrix=confusion_matrix,
        parent_id=trainer.training.last_known_model.id,
        train_image_count=trainer.training.data.train_image_count(),
        test_image_count=trainer.training.data.test_image_count(),
        trainer_id=trainer.uuid,
    )

    return new_model
