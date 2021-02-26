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


@pytest.fixture(autouse=True, scope='function')
def cleanup():
    shutil.rmtree('../data', ignore_errors=True)


@pytest.fixture(autouse=True, scope='module')
def create_project():
    test_helper.LiveServerSession().delete(f"/api/zauberzeug/projects/pytest?keep_images=true")
    project_configuration = {'project_name': 'pytest', 'inbox': 0, 'annotate': 0, 'review': 0, 'complete': 2, 'image_style': 'beautiful',
                             'categories': 2, 'thumbs': False, 'tags': 0, 'trainings': 1, 'detections': 3, 'annotations': 0, 'skeleton': False}
    assert test_helper.LiveServerSession().post(f"/api/zauberzeug/projects/generator",
                                                json=project_configuration).status_code == 200
    yield
    test_helper.LiveServerSession().delete(f"/api/zauberzeug/projects/pytest?keep_images=true")


def get_data() -> dict:
    response = test_helper.LiveServerSession().get(f'api/zauberzeug/projects/pytest/data?state=complete&mode=boxes')
    assert response.status_code == 200
    return response.json()


def test_download_images(web: Session):
    def get_files_files():
        return [entry for entry in glob('../data/**/*', recursive=True) if os.path.isfile(entry)]

    assert len(get_files_files()) == 0
    data = get_data()
    image_folder = main._create_image_folder('zauberzeug', 'pytest')
    resources_ids = main._extract_ressoure_ids(data)

    main._download_images('backend', resources_ids, image_folder)
    assert len(get_files_files()) == 2


def test_yolo_box_creation(web: Session):
    def get_files_files():
        return [entry for entry in glob('../data/**/*', recursive=True) if os.path.isfile(entry)]

    assert len(get_files_files()) == 0
    image_folder = main._create_image_folder('zauberzeug', 'pytest')
    data = get_data()

    main._update_yolo_boxes(image_folder, data)
    assert len(get_files_files()) == 2

    first_image_id = data['images'][0]['id']
    with open(f'{image_folder}/{first_image_id}.txt', 'r') as f:
        yolo_content = f.read()
    print(yolo_content, flush=True)
    assert yolo_content == '''0 0.550000 0.547917 0.050000 0.057500
1 0.725000 0.836250 0.050000 0.057500
1 0.175000 0.143750 0.050000 0.057500'''
