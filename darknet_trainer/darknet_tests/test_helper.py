from requests import Session
from urllib.parse import urljoin
from glob import glob
import os
import main


class LiveServerSession(Session):
    """https://stackoverflow.com/a/51026159/364388"""

    def __init__(self, *args, **kwargs):
        super(LiveServerSession, self).__init__(*args, **kwargs)
        self.prefix_url = 'http://backend:80/api'

    def request(self, method, url, *args, **kwargs):
        url = urljoin(self.prefix_url, url)
        return super(LiveServerSession, self).request(method, url, *args, **kwargs)


def get_data() -> dict:
    response = LiveServerSession().get(f'api/zauberzeug/projects/pytest/data?state=complete&mode=boxes')
    assert response.status_code == 200
    return response.json()


def get_files_from_data_folder():
    return [entry for entry in glob('../data/**/*', recursive=True) if os.path.isfile(entry)]


def create_needed_folders():
    project_folder = main._create_project_folder('zauberzeug', 'pytest')
    image_folder = main._create_image_folder(project_folder)
    training_folder = main._create_training_folder(project_folder, 'some_uuid')

    return project_folder, image_folder, training_folder
