from learning_loop_node import node_helper
from learning_loop_node.trainer.training_data import TrainingData
from learning_loop_node.trainer.trainer import Trainer
from requests import Session
from urllib.parse import urljoin
from glob import glob
import os
import main
from node import Node
import node
from typing import List
import asyncio


class LiveServerSession(Session):
    """https://stackoverflow.com/a/51026159/364388"""

    def __init__(self, *args, **kwargs):
        super(LiveServerSession, self).__init__(*args, **kwargs)
        self.prefix_url = node.SERVER_BASE_URL_DEFAULT

    def request(self, method, url, *args, **kwargs):
        url = urljoin(self.prefix_url, url)
        return super(LiveServerSession, self).request(method, url, *args, **kwargs)


async def get_training_data(node: Node) -> TrainingData:
    data = get_data2()
    image_ids = data['image_ids']
    image_data = await node_helper.download_images_data(node.url, node.headers, node.organization, node.project, image_ids)
    training_data = TrainingData(image_data=image_data, box_categories=data['box_categories'])
    return training_data


def get_data2() -> dict:
    response = LiveServerSession().get(f'api/zauberzeug/projects/pytest/data/data2?state=complete&mode=boxes')
    assert response.status_code == 200
    return response.json()


async def download_images(training_data, image_folder):
    urls, ids = node_helper.create_resource_urls(main.node.url, 'zauberzeug', 'pytest', training_data.image_ids())
    loop = asyncio.get_event_loop()
    await node_helper.download_images(loop, urls, ids, {}, image_folder)


def get_files_from_data_folder():
    files = [entry for entry in glob('../data/**/*', recursive=True) if os.path.isfile(entry)]
    files.sort()
    return files


def create_needed_folders(training_uuid='some_uuid'):
    project_folder = Node.create_project_folder('zauberzeug', 'pytest')
    image_folder = Trainer.create_image_folder(project_folder)
    training_folder = Trainer.create_training_folder(project_folder, training_uuid)

    return project_folder, image_folder, training_folder


def assert_upload_model() -> str:
    data = [('files', open('darknet_tests/test_data/fake_weightfile.weights', 'rb')),
            ('files', open('darknet_tests/test_data/tiny_yolo.cfg', 'rb'))]
    upload_response = LiveServerSession().post(
        f'/api/zauberzeug/projects/pytest/models', files=data)
    assert upload_response.status_code == 200
    return upload_response.json()['id']
