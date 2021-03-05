import pytest
import main
import shutil
import pytest
import darknet_tests.test_helper as test_helper
import yolo_helper
import yolo_cfg_helper
from uuid import uuid4
import os
from glob import glob
import time
import subprocess


@pytest.fixture(autouse=True, scope='function')
def cleanup():
    shutil.rmtree('../data', ignore_errors=True)
    yolo_helper.kill_all_darknet_processes()
    yield
    yolo_helper.kill_all_darknet_processes()


@pytest.fixture(autouse=True, scope='module')
def create_project():
    test_helper.LiveServerSession().delete(f"/api/zauberzeug/projects/pytest?keep_images=true")
    project_configuration = {'project_name': 'pytest', 'inbox': 0, 'annotate': 0, 'review': 0, 'complete': 2, 'image_style': 'beautiful',
                             'categories': 2, 'thumbs': False, 'tags': 0, 'trainings': 1, 'detections': 3, 'annotations': 0, 'skeleton': False}
    assert test_helper.LiveServerSession().post(f"/api/zauberzeug/projects/generator",
                                                json=project_configuration).status_code == 200
    yield
    test_helper.LiveServerSession().delete(f"/api/zauberzeug/projects/pytest?keep_images=true")


def test_download_images():
    assert len(test_helper.get_files_from_data_folder()) == 0
    data = test_helper.get_data()
    _, image_folder, _ = test_helper.create_needed_folders()
    resources = main._extract_image_ressoures(data)
    ids = main._extract_image_ids(data)

    main._download_images('backend', zip(resources, ids), image_folder)
    assert len(test_helper.get_files_from_data_folder()) == 2


def test_yolo_box_creation():
    assert len(test_helper.get_files_from_data_folder()) == 0
    _, image_folder, trainings_folder = test_helper.create_needed_folders()
    data = test_helper.get_data()
    image_ids = main._extract_image_ids(data)
    image_folder_for_training = yolo_helper.create_image_links(trainings_folder, image_folder, image_ids)

    yolo_helper.update_yolo_boxes(image_folder_for_training, data)
    assert len(test_helper.get_files_from_data_folder()) == 2

    first_image_id = image_ids[0]
    with open(f'{image_folder_for_training}/{first_image_id}.txt', 'r') as f:
        yolo_content = f.read()
    print(yolo_content, flush=True)
    assert yolo_content == '''0 0.550000 0.547917 0.050000 0.057500
1 0.725000 0.836250 0.050000 0.057500
1 0.175000 0.143750 0.050000 0.057500'''


def test_create_names_file():
    assert len(test_helper.get_files_from_data_folder()) == 0
    _, _, trainings_folder = test_helper.create_needed_folders()

    yolo_helper.create_names_file(trainings_folder, ['category_1', 'category_2'])
    files = test_helper.get_files_from_data_folder()
    assert len(files) == 1
    names_file = files[0]
    assert names_file.endswith('names.txt')
    with open(f'{trainings_folder}/names.txt', 'r') as f:
        names = f.readlines()

    assert len(names) == 2
    assert names[0] == 'category_1\n'
    assert names[1] == 'category_2'


def test_create_data_file():
    assert len(test_helper.get_files_from_data_folder()) == 0
    _, _, trainings_folder = test_helper.create_needed_folders()

    yolo_helper.create_data_file(trainings_folder, 1)
    files = test_helper.get_files_from_data_folder()
    assert len(files) == 1
    data_file = files[0]
    assert data_file.endswith('data.txt')
    with open(f'{trainings_folder}/data.txt', 'r') as f:
        data = f.readlines()

    assert len(data) == 5
    assert data[0] == 'classes = 1\n'
    assert data[1] == 'train  = train.txt\n'
    assert data[2] == 'valid  = test.txt\n'
    assert data[3] == 'names = names.txt\n'
    assert data[4] == 'backup = backup/'


def test_create_image_links():
    assert len(test_helper.get_files_from_data_folder()) == 0
    _, image_folder, trainings_folder = test_helper.create_needed_folders()

    data = test_helper.get_data()
    image_ids = main._extract_image_ids(data)
    image_resources = main._extract_image_ressoures(data)
    main._download_images('backend', zip(image_resources, image_ids), image_folder)

    yolo_helper.create_image_links(trainings_folder, image_folder, image_ids)

    files = test_helper.get_files_from_data_folder()
    assert len(files) == 4
    assert files[0] == '../data/zauberzeug/pytest/images/04e9b13d-9f5b-02c5-af46-5bf40b1ca0a7.jpg'
    assert files[1] == '../data/zauberzeug/pytest/images/94d1c90f-9ea5-abda-2696-6ab322d1e243.jpg'
    assert files[2] == '../data/zauberzeug/pytest/trainings/some_uuid/images/04e9b13d-9f5b-02c5-af46-5bf40b1ca0a7.jpg'
    assert files[3] == '../data/zauberzeug/pytest/trainings/some_uuid/images/94d1c90f-9ea5-abda-2696-6ab322d1e243.jpg'


def test_create_train_and_test_file():
    assert len(test_helper.get_files_from_data_folder()) == 0
    _, image_folder, trainings_folder = test_helper.create_needed_folders()
    data = test_helper.get_data()
    image_ids = main._extract_image_ids(data)
    images_folder_for_training = yolo_helper.create_image_links(trainings_folder, image_folder, image_ids)

    yolo_helper.create_train_and_test_file(trainings_folder, images_folder_for_training, data['images'])
    files = test_helper.get_files_from_data_folder()
    assert len(files) == 2
    test_file = files[0]
    train_file = files[1]
    assert train_file.endswith('train.txt')
    assert test_file.endswith('test.txt')
    with open(f'{trainings_folder}/train.txt', 'r') as f:
        content = f.readlines()

    assert len(content) == 1
    assert content[0] == '/data/zauberzeug/pytest/trainings/some_uuid/images/04e9b13d-9f5b-02c5-af46-5bf40b1ca0a7.jpg\n'

    with open(f'{trainings_folder}/test.txt', 'r') as f:
        content = f.readlines()

    assert len(content) == 1
    assert content[0] == '/data/zauberzeug/pytest/trainings/some_uuid/images/94d1c90f-9ea5-abda-2696-6ab322d1e243.jpg\n'


def test_download_model():
    assert len(test_helper.get_files_from_data_folder()) == 0

    _, _, trainings_folder = test_helper.create_needed_folders()

    model_id = _assert_upload_model()

    main._download_model(trainings_folder, 'zauberzeug', 'pytest', model_id, 'backend')
    files = test_helper.get_files_from_data_folder()
    assert len(files) == 2

    assert files[0] == "../data/zauberzeug/pytest/trainings/some_uuid/tiny_yolo.cfg"
    assert files[1] == "../data/zauberzeug/pytest/trainings/some_uuid/weightfile.weights"


def test_replace_classes_and_filters():
    target_folder = '/tmp/classes_test'

    def get_yolo_lines():
        with open(f'{target_folder}/yolo.cfg', 'r') as f:
            return f.readlines()

    def assert_line_count(search_string, expected_count):
        matched_lines = [line for line in get_yolo_lines() if line.strip() == search_string]
        assert len(matched_lines) == expected_count

    shutil.rmtree(target_folder, ignore_errors=True)
    os.makedirs(target_folder)

    shutil.copy('darknet_tests/test_data/tiny_yolo.cfg', f'{target_folder}/yolo.cfg')

    assert_line_count('filters=45', 0)
    assert_line_count('classes=10', 0)

    yolo_cfg_helper.replace_classes_and_filters(10, target_folder)

    assert_line_count('filters=45', 2)
    assert_line_count('classes=10', 2)


def test_create_anchors():

    model_id = _assert_upload_model()

    main.node.status.model = {'id': model_id}
    main.node.status.organization = 'zauberzeug'
    main.node.status.project = 'pytest'

    data = test_helper.get_data()
    training_uuid = str(uuid4())
    main._prepare_training(main.node, data, training_uuid)

    anchor_line = 'anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319'
    original_cfg_file_path = yolo_cfg_helper._find_cfg_file('darknet_tests/test_data')
    _assert_anchors(original_cfg_file_path, anchor_line)

    new_anchors = 'anchors=1.6000,1.8400,1.6000,1.8400,1.6000,1.8400,1.6000,1.8400,1.6000,1.8400,1.6000,1.8400'
    cfg_file_path = yolo_cfg_helper._find_cfg_file(f'../data/zauberzeug/pytest/trainings/{training_uuid}')
    _assert_anchors(cfg_file_path, new_anchors)


def test_start_training():

    model_id = _assert_upload_model()

    main.node.status.model = {'id': model_id}
    main.node.status.organization = 'zauberzeug'
    main.node.status.project = 'pytest'

    data = test_helper.get_data()
    training_uuid = str(uuid4())
    training_folder = main._prepare_training(main.node, data, training_uuid)
    assert training_folder == f'/data/zauberzeug/pytest/trainings/{training_uuid}'

    main._start_training(training_folder)
    # NOTE: /proc/{pid}/comm needs some time to show correct process name
    time.sleep(1)
    assert _is_any_darknet_running() == True, 'Training is not running'
    assert _wait_until_first_iteration_reached(training_folder, timeout=40) == True, 'No iteration created.'
    main._stop_training(training_folder)
    time.sleep(1)
    assert _is_any_darknet_running() == False


def _is_any_darknet_running():
    p = subprocess.Popen('pgrep darknet', shell=True)
    p.communicate()
    return p.returncode == 0


def _wait_until_first_iteration_reached(training_folder: str, timeout: int) -> bool:
    for i in range(timeout + 1):
        with open(f'{training_folder}/last_training.log', 'r') as f:
            content = f.read()

        if '(next mAP calculation at 1000 iterations)' in content:
            return True
        if not _is_any_darknet_running():
            print('Darknet issnt running anymore. Therefore the log issnt written anymore. Returning False.')
            return False
        time.sleep(1)
    return False


def _assert_anchors(cfg_file_path: str, anchor_line: str) -> None:
    anchor_line = anchor_line.replace(' ', '')
    found_anchor_line_count = 0
    with open(cfg_file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.replace(' ', '').strip()
        if line.startswith('anchors='):
            assert line == anchor_line, 'Anchor line does not match. '
            found_anchor_line_count += 1
    assert found_anchor_line_count > 0, 'There must be at least one anchorline in cfg file.'


def _assert_upload_model() -> str:

    data = [('files', open('darknet_tests/test_data/weightfile.weights', 'rb')),
            ('files', open('darknet_tests/test_data/tiny_yolo.cfg', 'rb'))]
    upload_response = test_helper.LiveServerSession().post(
        f'/api/zauberzeug/projects/pytest/models', files=data)
    assert upload_response.status_code == 200
    return upload_response.json()['url'].split('/')[-2]
