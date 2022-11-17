
from learning_loop_node.trainer.training import TrainingOut
from learning_loop_node.socket_response import SocketResponse
from fastapi.encoders import jsonable_encoder
import logging
import socketio
import asyncio


async def try_sync_model(trainer: any, trainer_node_uuid: str, sio_client: socketio.AsyncClient):
    try:
        model = trainer.get_new_model()
    except Exception as e:
        logging.exception('error while getting new model')
        raise Exception(f'Could not get new model : {str(e)}')
    logging.debug(f'new model {model}')

    if model:
        response = await sync_model(trainer, trainer_node_uuid, sio_client, model)

        if not response.success:
            error_msg = f'Error for update_training: Response from loop was : {response.__dict__}'
            logging.error(error_msg)
            raise Exception(error_msg)


async def sync_model(trainer, trainer_node_uuid, sio_client, model):
    current_training = trainer.training
    new_training = TrainingOut(
        trainer_id=trainer_node_uuid,
        confusion_matrix=model.confusion_matrix,
        train_image_count=current_training.data.train_image_count(),
        test_image_count=current_training.data.test_image_count(),
        hyperparameters=trainer.hyperparameters)

    await asyncio.sleep(0.1)  # NOTE needed for tests.

    result = await sio_client.call('update_training', (current_training.context.organization, current_training.context.project, jsonable_encoder(new_training)))
    response = SocketResponse.from_dict(result)

    if response.success:
        logging.info(f'successfully updated training {jsonable_encoder(new_training)}')
        trainer.on_model_published(model)
    return response
