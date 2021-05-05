from learning_loop_node.tests.test_helper import LiveServerSession
from learning_loop_node import node_helper
from learning_loop_node.trainer.training_data import TrainingData
from glob import glob
import os
import main
from node import Node
import asyncio


# async def get_training_data(node: Node) -> TrainingData:
#     data = get_data2()
#     image_ids = data['image_ids']
#     image_data = await node_helper.download_images_data(node.url, node.headers, node.organization, node.project, image_ids)
#     training_data = TrainingData(image_data=image_data, box_categories=data['box_categories'])
#     return training_data


# def get_data2() -> dict:
#     response = LiveServerSession().get(f'api/zauberzeug/projects/pytest/data/data2?state=complete&mode=boxes')
#     assert response.status_code == 200
#     return response.json()


# async def download_images(training_data, image_folder):
#     urls, ids = node_helper.create_resource_urls(main.node.url, 'zauberzeug', 'pytest', training_data.image_ids())
#     loop = asyncio.get_event_loop()
#     await node_helper.download_images(loop, urls, ids, {}, image_folder)


def get_files_from_data_folder():
    files = [entry for entry in glob('../data/**/*', recursive=True) if os.path.isfile(entry)]
    files.sort()
    return files
