from learning_loop_node.trainer.trainer import Trainer
from learning_loop_node import node
from learning_loop_node.trainer.downloader_factory import DownloaderFactory
from learning_loop_node.trainer.downloader import Downloader
from darknet_trainer import DarknetTrainer
from learning_loop_node.context import Context
import pytest
from typing import Generator
import learning_loop_node.tests.test_helper as test_helper
import learning_loop_node.trainer.tests.trainer_test_helper as trainer_test_helper
import darknet_tests.test_helper as darknet_test_helper

import shutil
import os


@pytest.fixture()
def web() -> Generator:
    with test_helper.LiveServerSession() as c:
        yield c


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
async def test_start_stop_training():
    model_id = trainer_test_helper.assert_upload_model(
        ['darknet_tests/test_data/tiny_yolo.cfg', 'darknet_tests/test_data/fake_weightfile.weights'])

    darknet_trainer = darknet_test_helper.create_darknet_trainer()
    downloader = darknet_test_helper.create_downloader()

    assert darknet_trainer.is_training_alive() == False
    context = Context(organization='zauberzeug', project='pytest')
    await darknet_trainer.begin_training(context=context, source_model={'id': model_id}, downloader=downloader)
    assert darknet_trainer.is_training_alive() == True

    darknet_trainer.stop_training()
    assert darknet_trainer.is_training_alive() == False


@pytest.mark.asyncio
async def test_get_model_files():
    darknet_trainer = darknet_test_helper.create_darknet_trainer()
    await darknet_test_helper.downlaod_data(darknet_trainer)

    shutil.copy('darknet_tests/test_data/fake_weightfile.weights',
                f'{darknet_trainer.training.training_folder}/some_model_uuid.weights')

    files = darknet_trainer.get_model_files('some_model_uuid')

    assert len(files) == 3
    assert 'some_model_uuid.weights' in files[0]
    assert 'tiny_yolo.cfg' in files[1]
    assert 'names.txt' in files[2]


@pytest.mark.asyncio
async def test_get_new_model():
    darknet_trainer = darknet_test_helper.create_darknet_trainer()
    await darknet_test_helper.downlaod_data(darknet_trainer)

    path = f'{darknet_trainer.training.training_folder}/backup/'
    os.makedirs(path)
    open(f'{path}/tiny_yolo_best_mAP_0.000000_iteration_1089_avgloss_-nan_.weights', 'a').close()

    shutil.copy('darknet_tests/test_data/last_training.log',
                f'{darknet_trainer.training.training_folder}/last_training.log')

    model = darknet_trainer.get_new_model()
    assert model is not None
