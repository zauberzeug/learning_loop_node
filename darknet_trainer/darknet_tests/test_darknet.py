import pytest
from requests import Session
from glob import glob
import main
import requests
import shutil
import os
import pytest
from requests import Session
from urllib.parse import urljoin
import darknet_tests.test_helper as test_helper


@pytest.fixture(autouse=True, scope='module')
def create_project():
    test_helper.LiveServerSession().delete(f"/api/zauberzeug/projects/pytest?keep_images=true")
    shutil.rmtree('../data', ignore_errors=True)
    project_configuration = {'project_name': 'pytest', 'inbox': 0, 'annotate': 0, 'review': 0, 'complete': 2, 'image_style': 'beautiful',
                             'categories': 2, 'thumbs': False, 'tags': 0, 'trainings': 1, 'detections': 3, 'annotations': 0, 'skeleton': False}
    assert test_helper.LiveServerSession().post(f"/api/zauberzeug/projects/generator",
                                                json=project_configuration).status_code == 200
    yield
    test_helper.LiveServerSession().delete(f"/api/zauberzeug/projects/pytest?keep_images=true")


def test_download_images(web: Session):
    def get_image_files():
        return [entry for entry in glob('../data/**/*', recursive=True) if os.path.isfile(entry)]

    assert len(get_image_files()) == 0
    hostname = 'backend'
    data = requests.get(f'http://backend/api/zauberzeug/projects/pytest/data?state=complete&mode=boxes').json()
    image_folder = main._create_image_folder('zauberzeug', 'pytest')
    resources_ids = main._extract_ressoure_ids(data)

    main._download_images(hostname, resources_ids, image_folder)
    assert len(get_image_files()) == 2
