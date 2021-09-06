from learning_loop_node.context import Context
from typing import List, Optional
from learning_loop_node.node import Node
from learning_loop_node.trainer.trainer import Trainer
import aiohttp
from learning_loop_node import loop
import logging


async def assert_upload_model(file_paths: Optional[List[str]] = None, format: str = 'mocked') -> str:
    if not file_paths:
        file_paths = ['trainer/tests/test_data/file_1.txt',
                      'trainer/tests/test_data/file_2.txt']
    data = [('files', open(path, 'rb')) for path in file_paths]

    data = aiohttp.FormData()

    for path in file_paths:
        data.add_field('files',  open(path, 'rb'))
    async with loop.post(f'api/zauberzeug/projects/pytest/models/{format}', data) as response:
        if response.status != 200:
            msg = f'unexpected status code {response.status} while posting new model'
            logging.error(msg)
            raise(Exception(msg))
        model = await response.json()
        return model['id']


def create_needed_folders(base_folder: str, training_uuid: str = 'some_uuid'):
    project_folder = Node.create_project_folder(
        Context(organization='zauberzeug', project='pytest', base_folder=base_folder))
    image_folder = Trainer.create_image_folder(project_folder)
    training_folder = Trainer.create_training_folder(project_folder, training_uuid)
    return project_folder, image_folder, training_folder
