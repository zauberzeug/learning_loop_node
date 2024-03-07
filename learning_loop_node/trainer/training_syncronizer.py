
import asyncio
import logging
from dataclasses import asdict
from typing import TYPE_CHECKING

import socketio
from dacite import from_dict
from fastapi.encoders import jsonable_encoder

from ..data_classes import TrainingOut
from ..data_classes.socket_response import SocketResponse

if TYPE_CHECKING:
    from .trainer_logic import TrainerLogic


class TrainingSyncronizer:
    def __init__(self, trainer_node_uuid: str, sio_client: socketio.AsyncClient):
        self.trainer_node_uuid = trainer_node_uuid
        self.sio_client = sio_client

    async def sync_model(model, current_training):
        new_training = TrainingOut(
            trainer_id=self.trainer_node_uuid,
            confusion_matrix=model.confusion_matrix,
            train_image_count=current_training.data.train_image_count(),
            test_image_count=current_training.data.test_image_count(),
            hyperparameters=trainer.hyperparameters)

        await asyncio.sleep(0.1)  # NOTE needed for tests.

        result = await self.sio_client.call('update_training', (current_training.context.organization, current_training.context.project, jsonable_encoder(new_training)))
        response = from_dict(data_class=SocketResponse, data=result)

        return response


async def try_sync_model(mo):
    try:
        model = trainer.get_new_model()
    except Exception as exc:
        logging.exception('error while getting new model')
        raise Exception(f'Could not get new model: {str(exc)}') from exc
    logging.debug(f'new model {model}')

    if model:
        response = await sync_model(trainer, trainer_node_uuid, sio_client, model)

        if not response.success:
            error_msg = f'Error for update_training: Response from loop was : {asdict(response)}'
            logging.error(error_msg)
            raise Exception(error_msg)
