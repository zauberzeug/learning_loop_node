from typing import List, Optional
from learning_loop_node.tests.test_helper import LiveServerSession
from learning_loop_node.node import Node
from learning_loop_node.trainer.trainer import Trainer


def assert_upload_model(file_paths: Optional[List[str]] = None) -> str:
    if not file_paths:
        file_paths = ['learning_loop_node/trainer/tests/test_data/file_1.txt',
                      'learning_loop_node/trainer/tests/test_data/file_2.txt']
    data = [('files', open(path, 'rb')) for path in file_paths]

    upload_response = LiveServerSession().post(
        f'/api/zauberzeug/projects/pytest/models', files=data)
    assert upload_response.status_code == 200
    return upload_response.json()['id']


def create_needed_folders(training_uuid='some_uuid'):
    project_folder = Node.create_project_folder('zauberzeug', 'pytest')
    image_folder = Trainer.create_image_folder(project_folder)
    training_folder = Trainer.create_training_folder(project_folder, training_uuid)
    return project_folder, image_folder, training_folder
