from learning_loop_node.trainer.trainer import Trainer
from learning_loop_node.trainer.training import TrainingOut
from learning_loop_node.socket_response import SocketResponse
from fastapi.encoders import jsonable_encoder
import logging


async def try_sync_model(trainer: Trainer):
    model = trainer.get_new_model()
    logging.debug(f'new model {model}')

    sio_client = None
    if model:
        response = await sync_model(trainer, sio_client, model)

        if not response.success:
            error_msg = f'Error for update_training: Response from loop was : {response.__dict__}'
            logging.error(error_msg)
            raise Exception(error_msg)


async def sync_model(trainer, sio_client, model):
    current_training = trainer.training
    new_training = TrainingOut(
        trainer_id=trainer.uuid,
        confusion_matrix=model.confusion_matrix,
        train_image_count=current_training.data.train_image_count(),
        test_image_count=current_training.data.test_image_count(),
        hyperparameters=trainer.hyperparameters)

    result = await sio_client.call('update_training', (current_training.context.organization, current_training.context.project, jsonable_encoder(new_training)))
    response = SocketResponse.from_dict(result)

    if response.success:
        logging.info(f'successfully updated training {jsonable_encoder(new_training)}')
        trainer.on_model_published(model)
