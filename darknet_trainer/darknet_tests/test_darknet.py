import pytest
import shutil
import pytest
import darknet_tests.test_helper as test_helper
import yolo_helper
import yolo_cfg_helper
import os
import learning_loop_node.trainer.tests.trainer_test_helper as trainer_test_helper
import helper
import darknet_tests.test_helper as darknet_test_helper


@pytest.mark.asyncio
async def test_yolo_box_creation():
    darknet_trainer = darknet_test_helper.create_darknet_trainer()
    await darknet_test_helper.downlaod_data(darknet_trainer)
    training = darknet_trainer.training
    training_data = training.data

    image_folder_for_training = yolo_helper.create_image_links(
        training. training_folder, training.images_folder, training_data.image_ids())

    await yolo_helper.update_yolo_boxes(image_folder_for_training,  training_data)
    assert len(test_helper.get_files_from_data_folder()) == 11  # 3 images, 3 image_links, 3 txt files, .cfg, .weights

    first_image_id = training_data.image_ids()[0]
    with open(f'{image_folder_for_training}/{first_image_id}.txt', 'r') as f:
        yolo_content = f.read()

    assert yolo_content == '''0 0.525000 0.836250 0.050000 0.057500
1 0.725000 0.490417 0.050000 0.057500
1 0.100000 0.894583 0.050000 0.057500'''


def test_create_names_file():
    assert len(test_helper.get_files_from_data_folder()) == 0
    _, _, training_folder = trainer_test_helper.create_needed_folders()

    yolo_helper.create_names_file(training_folder, ['category_1', 'category_2'])
    files = test_helper.get_files_from_data_folder()
    assert len(files) == 1
    names_file = files[0]
    assert names_file.endswith('names.txt')
    with open(f'{training_folder}/names.txt', 'r') as f:
        names = f.readlines()

    assert len(names) == 2
    assert names[0] == 'category_1\n'
    assert names[1] == 'category_2'


def test_create_data_file():
    assert len(test_helper.get_files_from_data_folder()) == 0
    _, _, training_folder = trainer_test_helper.create_needed_folders()

    yolo_helper.create_data_file(training_folder, 1)
    files = test_helper.get_files_from_data_folder()
    assert len(files) == 1
    data_file = files[0]
    assert data_file.endswith('data.txt')
    with open(f'{training_folder}/data.txt', 'r') as f:
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

    darknet_trainer = darknet_test_helper.create_darknet_trainer()
    await darknet_test_helper.downlaod_data(darknet_trainer)
    training = darknet_trainer.training
    training_data = training.data
    training_id = training.id

    yolo_helper.create_image_links(training.training_folder, training.images_folder, training_data.image_ids())

    files = test_helper.get_files_from_data_folder()
    assert len(files) == 8
    assert files[0] == '../data/zauberzeug/pytest/images/04e9b13d-9f5b-02c5-af46-5bf40b1ca0a7.jpg'
    assert files[1] == '../data/zauberzeug/pytest/images/94d1c90f-9ea5-abda-2696-6ab322d1e243.jpg'
    assert files[2] == '../data/zauberzeug/pytest/images/d99747e9-7c6f-5753-2769-4184f870f18b.jpg'
    assert files[3] == f'../data/zauberzeug/pytest/trainings/{training_id}/fake_weightfile.weights'
    assert files[4] == f'../data/zauberzeug/pytest/trainings/{training_id}/images/04e9b13d-9f5b-02c5-af46-5bf40b1ca0a7.jpg'
    assert files[5] == f'../data/zauberzeug/pytest/trainings/{training_id}/images/94d1c90f-9ea5-abda-2696-6ab322d1e243.jpg'
    assert files[6] == f'../data/zauberzeug/pytest/trainings/{training_id}/images/d99747e9-7c6f-5753-2769-4184f870f18b.jpg'
    assert files[7] == f'../data/zauberzeug/pytest/trainings/{training_id}/tiny_yolo.cfg'


@pytest.mark.asyncio
async def test_create_train_and_test_file():
    assert len(test_helper.get_files_from_data_folder()) == 0
    darknet_trainer = darknet_test_helper.create_darknet_trainer()
    await darknet_test_helper.downlaod_data(darknet_trainer)
    training = darknet_trainer.training
    training_data = training.data

    images_folder_for_training = yolo_helper.create_image_links(
        training.  training_folder, training.images_folder, training_data.image_ids())

    yolo_helper.create_train_and_test_file(
        training.training_folder, images_folder_for_training, training_data.image_data)

    files = [file for file in test_helper.get_files_from_data_folder() if file.endswith('test.txt')
             or file.endswith('train.txt')]
    assert len(files) == 2
    test_file = files[0]
    train_file = files[1]
    assert train_file.endswith('train.txt')
    assert test_file.endswith('test.txt')
    with open(f'{training.training_folder}/train.txt', 'r') as f:
        content = f.readlines()

    assert len(content) == 2
    assert content[0] == f'{training.training_folder}/images/04e9b13d-9f5b-02c5-af46-5bf40b1ca0a7.jpg\n'
    assert content[1] == f'{training.training_folder}/images/d99747e9-7c6f-5753-2769-4184f870f18b.jpg\n'

    with open(f'{training.training_folder}/test.txt', 'r') as f:
        content = f.readlines()

    assert len(content) == 1
    assert content[0] == f'{training.training_folder}/images/94d1c90f-9ea5-abda-2696-6ab322d1e243.jpg\n'


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
    assert len(test_helper.get_files_from_data_folder()) == 0

    darknet_trainer = darknet_test_helper.create_darknet_trainer()
    await darknet_test_helper.downlaod_data(darknet_trainer)
    training = darknet_trainer.training
    training_data = training.data

    image_folder_for_training = yolo_helper.create_image_links(
        training.  training_folder, training.images_folder, training_data.image_ids())
    await yolo_helper.update_yolo_boxes(image_folder_for_training, training_data)
    box_category_names = helper.get_box_category_names(training_data)
    yolo_helper.create_names_file(training.training_folder, box_category_names)
    yolo_helper.create_data_file(training.training_folder, len(box_category_names))
    yolo_helper.create_train_and_test_file(
        training. training_folder, image_folder_for_training, training_data.image_data)
    yolo_cfg_helper.update_anchors(training.training_folder)

    anchor_line = 'anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319'
    original_cfg_file_path = yolo_cfg_helper._find_cfg_file('darknet_tests/test_data')
    _assert_anchors(original_cfg_file_path, anchor_line)

    new_anchors = 'anchors=1.6000,1.8400,1.6000,1.8400,1.6000,1.8400,1.6000,1.8400,1.6000,1.8400,1.6000,1.8400'
    cfg_file_path = yolo_cfg_helper._find_cfg_file(training.training_folder)
    _assert_anchors(cfg_file_path, new_anchors)


@pytest.mark.parametrize("target_cfg_file", [
    ('some_file.cfg'),
    ('different_name.cfg'),
])
def test_find_cfg_file(target_cfg_file):
    _, _, training_path = trainer_test_helper.create_needed_folders()

    shutil.copy(f'darknet_tests/test_data/tiny_yolo.cfg', f'{training_path}/{target_cfg_file}')
    found_cfg_file = yolo_cfg_helper._find_cfg_file(training_path)
    assert target_cfg_file in found_cfg_file


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
