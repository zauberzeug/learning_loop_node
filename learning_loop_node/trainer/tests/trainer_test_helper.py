from typing import List, Optional
from learning_loop_node.tests.test_helper import LiveServerSession
from learning_loop_node.node import Node
from learning_loop_node.trainer.trainer import Trainer
import aiohttp
import node


async def assert_upload_model(file_paths: Optional[List[str]] = None) -> str:
    if not file_paths:
        file_paths = ['learning_loop_node/trainer/tests/test_data/file_1.txt',
                      'learning_loop_node/trainer/tests/test_data/file_2.txt']
    data = [('files', open(path, 'rb')) for path in file_paths]

    data = aiohttp.FormData()

    for path in file_paths:
        data.add_field('files',  open(path, 'rb'))

    async with aiohttp.ClientSession() as session:
        async with session.post(f'{node.SERVER_BASE_URL_DEFAULT}/api/zauberzeug/projects/pytest/models/mocked', data=data, ) as upload_response:
            assert upload_response.status == 200
            return (await upload_response.json())['id']


def create_needed_folders(training_uuid='some_uuid'):
    project_folder = Node.create_project_folder('zauberzeug', 'pytest')
    image_folder = Trainer.create_image_folder(project_folder)
    training_folder = Trainer.create_training_folder(project_folder, training_uuid)
    return project_folder, image_folder, training_folder
