from learning_loop_node.trainer.model import Model
import uuid
import random
from learning_loop_node.trainer.trainer import Training, Trainer, TrainingData
from fastapi.encoders import jsonable_encoder


async def increment_time(trainer: Trainer):

    if not trainer.training or not trainer.training.data:
        raise Exception("Could not imcrement time.")

    trainer.status.uptime = trainer.status.uptime + 5
    print('---- time', trainer.status.uptime, flush=True)
    confusion_matrix = {}
    for category in trainer.training.data.box_categories:
        try:
            # todo
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

    result = await trainer.sio.call('update_model', (trainer.training.organization, trainer.training.project, jsonable_encoder(new_model)))
    if result != True:
        raise Exception('could not update model: ' + str(result))
    trainer.training.last_produced_model = new_model
    from learning_loop_node.status import State
    await trainer.update_state(State.Running)
    return new_model
