from typing import List, Union
import aiohttp
from learning_loop_node.context import Context
from learning_loop_node.loop import loop
import logging
from icecream import ic


async def upload_model(context: Context, files: List[str], model_id: str, format: str) -> None:
    data = aiohttp.FormData()

    for file_name in files:
        data.add_field('files',  open(file_name, 'rb'))
    async with loop.put(f'api/{context.organization}/projects/{context.project}/models/{model_id}/{format}/file', data=data) as response:
        if response.status != 200:
            msg = f'---- could not upload model with id {model_id} and format {format}. Details: {await response.text()}'
            raise Exception(msg)
        else:
            logging.info(f'---- uploaded model with id {model_id} and format {format}.')


async def upload_model_for_training(context: Context, files: List[str], training_number: int, format: str) -> Union[dict, None]:
    data = aiohttp.FormData()

    for file_name in files:
        data.add_field('files',  open(file_name, 'rb'))
    async with loop.put(f'api/{context.organization}/projects/{context.project}/trainings/{training_number}/models/latest/{format}/file', data=data) as response:
        if response.status != 200:
            msg = f'---- could not upload model for training {training_number} and format {format}. Details: {await response.text()}'
            logging.error(msg)
            response.raise_for_status()
        else:
            uploaded_model = await response.json()
            logging.info(
                f'---- uploaded model for training {training_number} and format {format}. Model id is {uploaded_model}')
            return uploaded_model
