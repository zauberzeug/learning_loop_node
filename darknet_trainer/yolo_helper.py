import aiofiles
from learning_loop_node.training_data import TrainingData
import subprocess
from typing import List
import helper
import os
from glob import glob
from mAP_parser import MAPParser
import aiohttp
from node import Node
from retry import retry


def to_yolo(learning_loop_box, image_width, image_height, categories):
    w = float(learning_loop_box['width']) / float(image_width)
    h = float(learning_loop_box['height']) / float(image_height)
    x = (float((learning_loop_box['x']) + float(learning_loop_box['width']) / 2) / float(image_width))
    y = (float((learning_loop_box['y']) + float(learning_loop_box['height']) / 2) / float(image_height))

    yoloID = categories.index(learning_loop_box['category_id'])

    return ' '.join([
        str(yoloID),
        str("%.6f" % x),
        str("%.6f" % y),
        str("%.6f" % w),
        str("%.6f" % h)])


def create_data_file(training_folder: str, number_of_classes: int) -> None:
    number_of_classes = f'classes = {number_of_classes}'
    train = 'train  = train.txt'
    valid = 'valid  = test.txt'
    names = 'names = names.txt'
    backup = 'backup = backup/'
    with open(f'{training_folder}/data.txt', 'w') as f:
        data_object = [number_of_classes, train, valid, names, backup]
        f.write('\n'.join(data_object))


async def update_yolo_boxes(node: Node, image_folder_for_training: str, training_data: TrainingData) -> None:
    category_ids = helper.get_box_category_ids(training_data)

    for image in training_data.image_data:
        image_width, image_height = image['width'], image['height']
        image_id = image['id']

        yolo_boxes = []
        for box in image['box_annotations']:
            yolo_box = to_yolo(box, image_width, image_height, category_ids)
            yolo_boxes.append(yolo_box)

        async with aiofiles.open(f'{image_folder_for_training}/{image_id}.txt', 'w') as out_file:
            await out_file.write('\n'.join(yolo_boxes))


def create_names_file(training_folder: str, categories: List[str]) -> None:
    with open(f'{training_folder}/names.txt', 'w') as f:
        f.write('\n'.join(categories))


def create_image_links(training_folder: str, image_folder: str, image_ids: List[str]) -> str:
    training_images_path = f'{training_folder}/images'
    os.makedirs(training_images_path, exist_ok=True)
    for image_id in image_ids:
        source = os.path.join(image_folder, f'{image_id}.jpg')
        target = os.path.join(training_images_path, f'{image_id}.jpg')
        os.symlink(source, target)

    return training_images_path


def create_train_and_test_file(training_folder: str, image_folder_for_training: str, images: List[dict]) -> None:
    with open(f'{training_folder}/train.txt', 'w') as f:
        for image in images:
            if image['set'] == 'train':
                f.write(f"{image_folder_for_training}/{image['id']}.jpg\n")

    with open(f'{training_folder}/test.txt', 'w') as f:
        for image in images:
            if image['set'] == 'test':
                f.write(f"{image_folder_for_training}/{image['id']}.jpg\n")


def create_backup_dir(training_folder: str):
    backup_path = f'{training_folder}/backup'
    os.makedirs(backup_path, exist_ok=True)


def kill_all_darknet_processes():
    @retry(AssertionError, tries=5, delay=0.1)
    def assert_no_darknet_running():
        assert _is_any_darknet_running() == False
        return True

    p = subprocess.Popen('kill -9 `pgrep darknet`', shell=True)
    p.communicate()
    return p.returncode == 0 and assert_no_darknet_running()


def _is_any_darknet_running() -> bool:
    p = subprocess.Popen('pgrep darknet', shell=True)
    p.communicate()
    return p.returncode == 0


def parse_yolo_lines(lines: str, iteration: int = None) -> dict:

    parser = MAPParser(lines)
    data = parser.parse_mAP()
    data['classes'] = parser.parse_classes()

    if iteration:
        data['iteration'] = iteration
    else:
        data.update(parser.parse_training_status())

    # print(data)
    return data
