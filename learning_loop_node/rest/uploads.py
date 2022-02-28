from typing import List
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
            msg = f'---- could not upload model with id {model_id} and format {format}. Details: {response.content}'
            raise Exception(msg)
        else:
            logging.info(f'---- uploaded model with id {model_id} and format {format}.')
