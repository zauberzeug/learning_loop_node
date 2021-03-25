import asyncio
import shutil
from mAP_parser import MAPParser
from threading import Thread
import backdoor_controls
from fastapi_utils.tasks import repeat_every
import io
from learning_loop_node.node import Node
import learning_loop_node.node_helper as node_helper
import os
from typing import List, Union
import helper
import yolo_helper
import yolo_cfg_helper
from uuid import uuid4
import os
from glob import glob
import subprocess
from icecream import ic
import psutil
from status import Status
from uuid import uuid4
import traceback


node = Node(uuid='c34dc41f-9b76-4aa9-8b8d-9d27e33a19e4',
            name='darknet trainer')

# wie geht das hier überhaupt? das wird doch erst beim ausführen von begin_training getriggert oder nicht?


@node.get_model_files
def get_model_files(organization: str, project: str, model_id: str) -> List[str]:
    return _get_model_files(model_id)


def _get_model_files(model_id: str) -> List[str]:
    try:
        weightfile_path = glob(f'/data/**/trainings/**/{model_id}.weights', recursive=True)[0]
    except:
        raise Exception(f'No model found for id: {model_id}.')

    training_path = '/'.join(weightfile_path.split('/')[:-1])
    cfg_file_path = glob(f'{training_path}/*.cfg', recursive=True)[0]
    return [weightfile_path, cfg_file_path, f'{training_path}/names.txt']


@node.begin_training
def begin_training(data: dict) -> None:
    training_uuid = str(uuid4())
    _prepare_training(node, data, training_uuid)
    _start_training(training_uuid)


def _prepare_training(node: Node, data: dict, training_uuid: str) -> None:
    _set_node_properties(node, data)
    project_folder = _create_project_folder(
        node.status.organization, node.status.project)
    image_folder = _create_image_folder(project_folder)
    image_resources = _extract_image_ressoures(data)
    image_ids = _extract_image_ids(data)
    node_helper.download_images(node, zip(
        image_resources, image_ids), image_folder)

    training_folder = _create_training_folder(project_folder, training_uuid)
    yolo_helper.create_backup_dir(training_folder)

    image_folder_for_training = yolo_helper.create_image_links(
        training_folder, image_folder, image_ids)

    yolo_helper.update_yolo_boxes(image_folder_for_training, data)

    box_category_names = helper._get_box_category_names(data)
    box_category_count = len(box_category_names)
    yolo_helper.create_names_file(training_folder, box_category_names)
    yolo_helper.create_data_file(training_folder, box_category_count)
    yolo_helper.create_train_and_test_file(
        training_folder, image_folder_for_training, data['images'])

    node_helper.download_model(node, training_folder, node.status.organization,
                               node.status.project, node.status.model['id'])
    yolo_cfg_helper.replace_classes_and_filters(
        box_category_count, training_folder)
    yolo_cfg_helper.update_anchors(training_folder)


def _set_node_properties(node: Node, data: dict) -> None:
    node.status.box_categories = data['box_categories']
    train_images, test_images = _get_train_and_test_images(data['images'])
    node.status.train_images = train_images
    node.status.test_images = test_images


def _get_train_and_test_images(images: dict) -> None:
    train_image_count = [image for image in images if image['set'] == 'train']
    test_image_count = [image for image in images if image['set'] == 'test']
    return train_image_count, test_image_count


def _create_project_folder(organization: str, project: str) -> str:
    project_folder = f'/data/{organization}/{project}'
    os.makedirs(project_folder, exist_ok=True)
    return project_folder


def _extract_image_ressoures(data: dict) -> List[tuple]:
    return [i['resource'] for i in data['images']]


def _extract_image_ids(data: dict) -> List[str]:
    return [i['id'] for i in data['images']]


def _create_image_folder(project_folder: str) -> str:
    image_folder = f'{project_folder}/images'
    os.makedirs(image_folder, exist_ok=True)
    return image_folder


def _create_training_folder(project_folder: str, trainings_id: str) -> str:
    training_folder = f'{project_folder}/trainings/{trainings_id}'
    os.makedirs(training_folder, exist_ok=True)
    return training_folder


def _start_training(training_id: str) -> None:
    training_path = get_training_path_by_id(training_id)

    weightfile = find_weightfile(training_path)
    cfg_file = find_cfg_file(training_path)
    # NOTE we have to write the pid inside the bash command to get the correct pid.
    cmd = f'cd {training_path};nohup /darknet/darknet detector train data.txt {cfg_file} {weightfile} -dont_show -map >> last_training.log 2>&1 & echo $! > last_training.pid'
    p = subprocess.Popen(cmd, shell=True)
    _, err = p.communicate()
    if p.returncode != 0:
        raise Exception(f'Failed to start training with error: {err}')

    node.status.model['training_id'] = training_id


def find_weightfile(training_path: str) -> str:
    weightfiles = glob(f'{training_path}/*.weights', recursive=True)
    if not weightfiles or len(weightfiles) > 1:
        raise Exception('Number of present weightfiles must be 1.')
    weightfile = weightfiles[0].split('/')[-1]
    if weightfile == 'fake_weightfile.weights':
        return ""
    else:
        return weightfile


def find_cfg_file(training_path: str) -> str:
    cfg_files = weightfiles = glob(f'{training_path}/*.cfg', recursive=True)
    if not weightfiles or len(weightfiles) > 1:
        raise Exception('Number of present .cfg files must be 1.')
    cfg_file = cfg_files[0].split('/')[-1]
    return cfg_file


@node.stop_training
def stop() -> None:
    _stop_training(node.status.model['training_id'])


def _stop_training(training_id: str) -> None:
    training_path = get_training_path_by_id(training_id)

    cmd = f'cd {training_path};kill -9 `cat last_training.pid`; rm last_training.pid'
    p = subprocess.Popen(cmd, shell=True)
    _, err = p.communicate()
    if p.returncode != 0:
        raise Exception(f'Failed to stop training with error: {err}')


@node.on_event("startup")
@repeat_every(seconds=5, raise_exceptions=False, wait_first=False)
async def check_state() -> None:
    """checking the current status"""
    try:
    await _check_state()
    except:
        traceback.print_exc()
        


async def _check_state() -> None:
    ic(f"checking state: {node.status.state}")
    if node.status.model:
        training_id = node.status.model['training_id']
        ic(training_id,)
        if training_id:

            await _check_for_new_model(training_id)
            await check_training_state(training_id)


async def check_training_state(training_id: str) -> None:
    state = get_training_state(training_id)
    ic(state, )
    if state == 'crashed':
        try:
            _stop_training(training_id)
        except:
            pass
        new_status = Status(id=node.status.id, name=node.status.name)
        await node.update_status(new_status)


async def _check_for_new_model(training_id: str) -> None:
    model = parse_latest_iteration(training_id)
    if model:
        last_published_iteration = node.status.model.get('last_published_iteration')
        if not last_published_iteration or model['iteration'] > last_published_iteration:
            model_id = str(uuid4())
            training_path = get_training_path_by_id(training_id)
            weightfile_name = model['weightfile']
            if weightfile_name:
                weightfile_path = f'{training_path}/{weightfile_name}'
                shutil.move(weightfile_path, f'{training_path}/{model_id}.weights')

            new_model = {
                'id': model_id,
                'hyperparameters': node.status.hyperparameters,
                'confusion_matrix': model['confusion_matrix'],
                'parent_id': node.status.model['id'],
                'train_image_count': len(node.status.train_images),
                'test_image_count': len(node.status.test_images),
                'trainer_id': node.status.id,
            }
            await node.sio.call('update_model', (node.status.organization, node.status.project, new_model))
            node.status.model.update(new_model)
            node.status.model['last_published_iteration'] = model['iteration']


def parse_latest_iteration(training_id: str) -> Union[dict, None]:
    training_path = get_training_path_by_id(training_id)
    log_file_path = f'{training_path}/last_training.log'

    with open(log_file_path, 'r') as f:
        log = f.read()

    iteration_log = MAPParser.extract_iteration_log(log)
    if not iteration_log:
        return None

    parser = MAPParser(iteration_log)
    iteration = parser.parse_iteration()

    confusion_matrices = {}
    for parsed_class in parser.parse_classes():
        name = parsed_class['name']
        id = _get_id_of_category_from_name(name)
        del parsed_class['id']
        del parsed_class['name']
        confusion_matrices[id] = parsed_class

    weightfile = parser.parse_weightfile()
    return {'iteration': iteration, 'confusion_matrix': confusion_matrices, 'weightfile': weightfile}


def _get_id_of_category_from_name(name: str) -> str:
    category_id = [category['id'] for category in node.status.box_categories if category['name'] == name]
    return category_id[0]


def get_training_state(training_id):
    training_path = get_training_path_by_id(training_id)
    pid_path = f'{training_path}/last_training.pid'
    if not os.path.exists(pid_path):
        return 'stopped'
    with open(pid_path, 'r') as f:
        pid = f.read().strip()
    try:
        p = psutil.Process(int(pid))
    except psutil.NoSuchProcess as e:
        return 'crashed'
    if p.name() == 'darknet':
        return 'running'
    return 'crashed'


def get_training_path_by_id(trainings_id: str) -> str:
    trainings = [training_path for training_path in glob(
        f'/data/**/trainings/{trainings_id}', recursive=True)]
    return trainings[0]


@node.on_event("shutdown")
async def shutdown():

    def restart():
        asyncio.create_task(node.sio.disconnect())

    Thread(target=restart).start()


# setting up backdoor_controls
node.include_router(backdoor_controls.router, prefix="")
