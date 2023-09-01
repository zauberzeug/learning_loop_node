import logging
from typing import List, Optional, Union

from learning_loop_node.data_classes import Context
from learning_loop_node.loop_communication import glc


async def upload_model(context: Context, files: List[str], model_id: str, mformat: str) -> None:
    response = await glc.put(f'/{context.organization}/projects/{context.project}/models/{model_id}/{mformat}/file', files=files)
    if response.status_code != 200:
        msg = f'---- could not upload model with id {model_id} and format {mformat}. Details: {response.text}'
        raise Exception(msg)
    logging.info(f'---- uploaded model with id {model_id} and format {mformat}.')


async def upload_model_for_training(context: Context, files: List[str], training_number: Optional[int], mformat: str) -> Union[dict, None]:
    response = await glc.put(f'/{context.organization}/projects/{context.project}/trainings/{training_number}/models/latest/{mformat}/file', files=files)
    if response.status_code != 200:
        msg = f'---- could not upload model for training {training_number} and format {mformat}. Details: {response.text}'
        logging.error(msg)
        response.raise_for_status()
    else:
        uploaded_model = response.json()
        logging.info(
            f'---- uploaded model for training {training_number} and format {mformat}. Model id is {uploaded_model}')
        return uploaded_model
