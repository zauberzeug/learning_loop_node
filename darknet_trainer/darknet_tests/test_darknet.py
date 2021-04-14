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
from icecream import ic
import learning_loop_node.node_helper as node_helper
from status import State
import asyncio


@pytest.fixture(autouse=True, scope='function')
def cleanup():

    shutil.rmtree('../data', ignore_errors=True)
    yolo_helper.kill_all_darknet_processes()
    yield
    yolo_helper.kill_all_darknet_processes()


@pytest.fixture(autouse=True, scope='module')
def create_project():
    test_helper.LiveServerSession().delete(f"/api/zauberzeug/projects/pytest?keep_images=true")
    project_configuration = {'project_name': 'pytest', 'inbox': 0, 'annotate': 0, 'review': 0, 'complete': 3, 'image_style': 'beautiful',
                             'categories': 2, 'thumbs': False, 'tags': 0, 'trainings': 1, 'detections': 3, 'annotations': 0, 'skeleton': False}
    assert test_helper.LiveServerSession().post(f"/api/zauberzeug/projects/generator",
                                                json=project_configuration).status_code == 200
    yield
    test_helper.LiveServerSession().delete(f"/api/zauberzeug/projects/pytest?keep_images=true")


@pytest.mark.asyncio
async def test_download_images():
    _, image_folder, _ = test_helper.create_needed_folders()
    assert len(test_helper.get_files_from_data_folder()) == 0

    training_data = await test_helper.get_training_data(main.node)
    assert len(training_data.image_data) == 3

    await node_helper.download_images(main.node.url, main.node.headers, training_data.image_data, image_folder)
    assert len(test_helper.get_files_from_data_folder()) == 3


@pytest.mark.asyncio
async def test_yolo_box_creation():
    assert len(test_helper.get_files_from_data_folder()) == 0
    _, image_folder, trainings_folder = test_helper.create_needed_folders()

    training_data = await test_helper.get_training_data(main.node)

    image_folder_for_training = yolo_helper.create_image_links(
        trainings_folder, image_folder, training_data.image_ids())

    await yolo_helper.update_yolo_boxes(main.node, image_folder_for_training,  training_data)
    assert len(test_helper.get_files_from_data_folder()) == 3

    first_image_id = training_data.image_ids()[0]
    with open(f'{image_folder_for_training}/{first_image_id}.txt', 'r') as f:
        yolo_content = f.read()
    print(yolo_content, flush=True)
    assert yolo_content == '''0 0.525000 0.836250 0.050000 0.057500
1 0.725000 0.490417 0.050000 0.057500
1 0.100000 0.894583 0.050000 0.057500'''


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


@pytest.mark.asyncio
async def test_create_image_links():
    assert len(test_helper.get_files_from_data_folder()) == 0
    _, image_folder, trainings_folder = test_helper.create_needed_folders()
    training_data = await test_helper.get_training_data(main.node)

    await node_helper.download_images(main.node.url, main.node.headers,  training_data.image_data, image_folder)

    yolo_helper.create_image_links(trainings_folder, image_folder, training_data.image_ids())

    files = test_helper.get_files_from_data_folder()
    assert len(files) == 6
    assert files[0] == '../data/zauberzeug/pytest/images/04e9b13d-9f5b-02c5-af46-5bf40b1ca0a7.jpg'
    assert files[1] == '../data/zauberzeug/pytest/images/94d1c90f-9ea5-abda-2696-6ab322d1e243.jpg'
    assert files[2] == '../data/zauberzeug/pytest/images/d99747e9-7c6f-5753-2769-4184f870f18b.jpg'
    assert files[3] == '../data/zauberzeug/pytest/trainings/some_uuid/images/04e9b13d-9f5b-02c5-af46-5bf40b1ca0a7.jpg'
    assert files[4] == '../data/zauberzeug/pytest/trainings/some_uuid/images/94d1c90f-9ea5-abda-2696-6ab322d1e243.jpg'
    assert files[5] == '../data/zauberzeug/pytest/trainings/some_uuid/images/d99747e9-7c6f-5753-2769-4184f870f18b.jpg'


@pytest.mark.asyncio
async def test_create_train_and_test_file():
    assert len(test_helper.get_files_from_data_folder()) == 0
    _, image_folder, trainings_folder = test_helper.create_needed_folders()

    training_data = await test_helper.get_training_data(main.node)
    assert len(training_data.image_data) == 3

    images_folder_for_training = yolo_helper.create_image_links(
        trainings_folder, image_folder, training_data.image_ids())

    yolo_helper.create_train_and_test_file(trainings_folder, images_folder_for_training, training_data.image_data)
    files = test_helper.get_files_from_data_folder()
    assert len(files) == 2
    test_file = files[0]
    train_file = files[1]
    assert train_file.endswith('train.txt')
    assert test_file.endswith('test.txt')
    with open(f'{trainings_folder}/train.txt', 'r') as f:
        content = f.readlines()

    assert len(content) == 2
    assert content[0] == '/data/zauberzeug/pytest/trainings/some_uuid/images/04e9b13d-9f5b-02c5-af46-5bf40b1ca0a7.jpg\n'
    assert content[1] == '/data/zauberzeug/pytest/trainings/some_uuid/images/d99747e9-7c6f-5753-2769-4184f870f18b.jpg\n'

    with open(f'{trainings_folder}/test.txt', 'r') as f:
        content = f.readlines()

    assert len(content) == 1
    assert content[0] == '/data/zauberzeug/pytest/trainings/some_uuid/images/94d1c90f-9ea5-abda-2696-6ab322d1e243.jpg\n'


def test_download_model():
    assert len(test_helper.get_files_from_data_folder()) == 0

    _, _, trainings_folder = test_helper.create_needed_folders()

    model_id = test_helper.assert_upload_model()

    node_helper.download_model(main.node.url, main.node.headers, trainings_folder, 'zauberzeug', 'pytest', model_id)
    files = test_helper.get_files_from_data_folder()
    assert len(files) == 2

    assert files[0] == "../data/zauberzeug/pytest/trainings/some_uuid/fake_weightfile.weights"
    assert files[1] == "../data/zauberzeug/pytest/trainings/some_uuid/tiny_yolo.cfg"


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


@pytest.mark.asyncio
async def test_create_anchors():
    model_id = test_helper.assert_upload_model()

    main.node.status.model = {'id': model_id}
    main.node.status.organization = 'zauberzeug'
    main.node.status.project = 'pytest'

    training_data = await test_helper.get_training_data(main.node)
    training_uuid = str(uuid4())
    await main._prepare_training(main.node, training_data, training_uuid)

    anchor_line = 'anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319'
    original_cfg_file_path = yolo_cfg_helper._find_cfg_file('darknet_tests/test_data')
    _assert_anchors(original_cfg_file_path, anchor_line)

    new_anchors = 'anchors=1.6000,1.8400,1.6000,1.8400,1.6000,1.8400,1.6000,1.8400,1.6000,1.8400,1.6000,1.8400'
    cfg_file_path = yolo_cfg_helper._find_cfg_file(f'../data/zauberzeug/pytest/trainings/{training_uuid}')
    _assert_anchors(cfg_file_path, new_anchors)


@pytest.mark.asyncio
async def test_start_training():
    training_uuid = str(uuid4())
    await _start_training(training_uuid)

    # NOTE: /proc/{pid}/comm needs some time to show correct process name
    await asyncio.sleep(1)

    assert _is_any_darknet_running() == True, 'Training is not running'
    assert _wait_until_first_iteration_reached(training_uuid, timeout=40) == True, 'No iteration created.'
    main._stop_training(training_uuid)
    time.sleep(1)
    assert _is_any_darknet_running() == False


@pytest.mark.asyncio
async def test_check_running_training_state():
    training_uuid = str(uuid4())
    await _start_training(training_uuid)

    state = main.get_training_state(training_uuid)
    assert state == 'running'


@pytest.mark.asyncio
async def test_check_crashed_training_state():
    training_uuid = str(uuid4())

    await _start_training(training_uuid)

    yolo_helper.kill_all_darknet_processes()
    time.sleep(.1)

    state = main.get_training_state(training_uuid)
    assert state == 'crashed'


@pytest.mark.asyncio
async def test_cleanup_after_crash():
    training_uuid = str(uuid4())
    await _start_training(training_uuid)
    assert main.node.status.model['training_id'] == training_uuid

    state = main.get_training_state(training_uuid)
    assert state == 'running'
    assert _pid_file_exists(training_uuid) == True
    yolo_helper.kill_all_darknet_processes()

    await main._check_state()
    assert _pid_file_exists(training_uuid) == False
    assert main.node.status.model == None


@pytest.mark.asyncio
async def test_reset_to_idle_after_crash():
    await main.node.connect()
    await main.node.sio.sleep(1.0)
    await main.node.update_state(State.Idle)
    _assert_trainer_state(State.Idle)

    model_id = test_helper.assert_upload_model()
    main.node.status.model = {'id': model_id}
    model = {'id': model_id}
    begin_training_handler = main.node.sio.handlers['/']['begin_training']

    await begin_training_handler('zauberzeug', 'pytest', model)
    await asyncio.sleep(3)  # TODO how to wait here?
    _assert_trainer_state(State.Running)

    yolo_helper.kill_all_darknet_processes()
    await main._check_state()
    _assert_trainer_state(State.Idle)


@pytest.mark.asyncio
async def test_get_files_for_model_id():
    training_uuid = uuid4()
    _, _, training_path = test_helper.create_needed_folders(training_uuid)
    new_model_id = uuid4()
    open(f'{training_path}/{new_model_id}.weights', 'a').close()
    open(f'{training_path}/my_yolo.cfg', 'a').close()
    open(f'{training_path}/names.txt', 'a').close()

    files = main._get_model_files(new_model_id)
    assert len(files) == 3
    assert files[0].split('/')[-1] == f'{new_model_id}.weights'
    assert files[1].split('/')[-1] == 'my_yolo.cfg'
    assert files[2].split('/')[-1] == 'names.txt'


def test_find_weightfile():
    training_uuid = uuid4()
    _, _, training_path = test_helper.create_needed_folders(training_uuid)
    shutil.copy('darknet_tests/test_data/fake_weightfile.weights', f'{training_path}/fake_weightfile.weights')

    weightfile = main.find_weightfile(training_path)
    assert weightfile == ''

    os.remove(f'{training_path}/fake_weightfile.weights')
    shutil.copy('darknet_tests/test_data/fake_weightfile.weights', f'{training_path}/weightfile.weights')

    weightfile = main.find_weightfile(training_path)
    assert weightfile == 'weightfile.weights'


@pytest.mark.parametrize("target_cfg_file", [
    ('some_file.cfg'),
    ('different_name.cfg'),
])
def test_find_cfg_file(target_cfg_file):
    _, _, training_path = test_helper.create_needed_folders(uuid4)

    shutil.copy(f'darknet_tests/test_data/tiny_yolo.cfg', f'{training_path}/{target_cfg_file}')
    cfg_file = main.find_cfg_file(training_path)
    assert cfg_file == target_cfg_file


async def _start_training(training_uuid):
    model_id = test_helper.assert_upload_model()
    main.node.status.model = {'id': model_id}
    main.node.status.organization = 'zauberzeug'
    main.node.status.project = 'pytest'
    data = test_helper.get_data2()
    training_data = await test_helper.get_training_data(main.node)
    await main._prepare_training(main.node, training_data, training_uuid)
    await main._start_training(training_uuid)


def _is_any_darknet_running() -> bool:
    p = subprocess.Popen('pgrep darknet', shell=True)
    p.communicate()
    return p.returncode == 0


def _wait_until_first_iteration_reached(training_uuid: str, timeout: int) -> bool:
    training_folder = main.helper.get_training_path_by_id(training_uuid)
    for i in range(timeout + 1):
        with open(f'{training_folder}/last_training.log', 'r') as f:
            content = f.read()

        if '(next mAP calculation at 1000 iterations)' in content:
            return True
        if not _is_any_darknet_running():
            log = None
            try:
                with open(f'{training_folder}/last_training.log', 'r') as f:
                    log = f.read()
            except:
                pass
            raise Exception(
                f'Darknet issnt running anymore. Therefore the log issnt written anymore. Returning False.\nLog: {log}')
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


def _pid_file_exists(training_uuid: str) -> bool:
    training_path = main.helper.get_training_path_by_id(training_uuid)
    pid_path = f'{training_path}/last_training.pid'
    return os.path.exists(pid_path)


def _assert_trainer_state(state: State) -> None:
    training_response = test_helper.LiveServerSession().get(
        f'/api/zauberzeug/projects/pytest/trainings')
    assert training_response.status_code == 200
    trainers = training_response.json()['trainers']
    darknet_trainer = [trainer for trainer in trainers if trainer['name'] == 'darknet trainer'][0]
    assert darknet_trainer['state'] == state
