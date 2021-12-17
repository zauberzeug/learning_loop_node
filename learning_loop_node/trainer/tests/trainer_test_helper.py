from learning_loop_node.context import Context
from learning_loop_node.node import Node
from learning_loop_node.trainer.trainer import Trainer
from learning_loop_node import node_helper


def create_needed_folders(base_folder: str, training_uuid: str = 'some_uuid'):
    project_folder = Node.create_project_folder(
        Context(organization='zauberzeug', project='pytest', base_folder=base_folder))
    image_folder = node_helper.create_image_folder(project_folder)
    training_folder = Trainer.create_training_folder(project_folder, training_uuid)
    return project_folder, image_folder, training_folder
