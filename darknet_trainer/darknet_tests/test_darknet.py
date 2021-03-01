import pytest
from requests import Session
from glob import glob
import main
import shutil
import os
import pytest
from requests import Session
from urllib.parse import urljoin
import darknet_tests.test_helper as test_helper
import yolo_helper


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


def get_files_from_data_folder():
    return [entry for entry in glob('../data/**/*', recursive=True) if os.path.isfile(entry)]


def test_download_images(web: Session):
    assert len(get_files_from_data_folder()) == 0
    data = get_data()
    project_folder = main._create_project_folder('zauberzeug', 'pytest')
    image_folder = main._create_image_folder(project_folder)
    resources = main._extract_image_ressoures(data)
    ids = main._extract_image_ids(data)

    main._download_images('backend', zip(resources, ids), image_folder)
    assert len(get_files_from_data_folder()) == 2


def test_yolo_box_creation(web: Session):
    assert len(get_files_from_data_folder()) == 0
    project_folder = main._create_project_folder('zauberzeug', 'pytest')
    image_folder = main._create_image_folder(project_folder)
    trainings_folder = main._create_trainings_folder(project_folder, 'some_uuid')
    data = get_data()
    image_ids = main._extract_image_ids(data)
    image_folder_for_training = yolo_helper.create_image_links(trainings_folder, image_folder, image_ids)

    yolo_helper.update_yolo_boxes(image_folder_for_training, data)
    assert len(get_files_from_data_folder()) == 2

    first_image_id = image_ids[0]
    with open(f'{image_folder_for_training}/{first_image_id}.txt', 'r') as f:
        yolo_content = f.read()
    print(yolo_content, flush=True)
    assert yolo_content == '''0 0.550000 0.547917 0.050000 0.057500
1 0.725000 0.836250 0.050000 0.057500
1 0.175000 0.143750 0.050000 0.057500'''


def test_create_names_file(web: Session):
    assert len(get_files_from_data_folder()) == 0
    project_folder = main._create_project_folder('zauberzeug', 'pytest')
    trainings_path = main._create_trainings_folder(project_folder, 'some_uuid')

    yolo_helper.create_names_file(trainings_path, ['category_1', 'category_2'])
    files = get_files_from_data_folder()
    assert len(files) == 1
    names_file = files[0]
    assert names_file.endswith('names.txt')
    with open(f'{trainings_path}/names.txt', 'r') as f:
        names = f.readlines()

    assert len(names) == 2
    assert names[0] == 'category_1\n'
    assert names[1] == 'category_2'


def test_create_data_file():
    assert len(get_files_from_data_folder()) == 0
    project_folder = main._create_project_folder('zauberzeug', 'pytest')
    trainings_path = main._create_trainings_folder(project_folder, 'some_uuid')

    yolo_helper.create_data_file(trainings_path, 1)
    files = get_files_from_data_folder()
    assert len(files) == 1
    data_file = files[0]
    assert data_file.endswith('data.txt')
    with open(f'{trainings_path}/data.txt', 'r') as f:
        data = f.readlines()

    assert len(data) == 5
    assert data[0] == 'classes = 1\n'
    assert data[1] == 'train  = train.txt\n'
    assert data[2] == 'valid  = test.txt\n'
    assert data[3] == 'names = names.txt\n'
    assert data[4] == 'backup = backup/'


def test_create_image_links():
    assert len(get_files_from_data_folder()) == 0
    project_folder = main._create_project_folder('zauberzeug', 'pytest')
    image_folder = main._create_image_folder(project_folder)
    trainings_path = main._create_trainings_folder(project_folder, 'some_uuid')

    data = get_data()
    image_ids = main._extract_image_ids(data)
    image_resources = main._extract_image_ressoures(data)
    yolo_helper.create_image_links(trainings_path, image_folder, image_ids)

    main._download_images('backend', zip(image_resources, image_ids), image_folder)
    files = get_files_from_data_folder()
    assert len(files) == 2
    assert files[0] == "../data/zauberzeug/pytest/images/04e9b13d-9f5b-02c5-af46-5bf40b1ca0a7.jpg"
    assert files[1] == "../data/zauberzeug/pytest/images/94d1c90f-9ea5-abda-2696-6ab322d1e243.jpg"
