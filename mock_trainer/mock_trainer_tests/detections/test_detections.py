
from glob import glob

import pytest

from learning_loop_node.data_classes import Context
from learning_loop_node.globals import GLOBALS
from learning_loop_node.loop_communication import LoopCommunicator
from learning_loop_node.tests import test_helper
from learning_loop_node.trainer.trainer import Trainer
from mock_trainer.mock_trainer import MockTrainer


@pytest.fixture()
async def setup_test_project(glc: LoopCommunicator):
    await glc.delete("/zauberzeug/projects/pytest?keep_images=true")
    project_configuration = {
        'project_name': 'pytest', 'inbox': 1, 'annotate': 2, 'review': 3, 'complete': 4, 'image_style': 'plain',
        'box_categories': 1, 'point_categories': 1, 'segmentation_categories': 1, 'thumbs': False, 'trainings': 1}
    for _ in range(10):  # NOTE this retry seems to fix huge flakyness in test runs on drone (local it's fine without)
        await glc.get("/status")
        response = await glc.post("/zauberzeug/projects/generator", json=project_configuration)
        if response.status_code == 200:
            break
    else:
        raise Exception('Could not create project')
    yield
    await glc.delete("/zauberzeug/projects/pytest?keep_images=true")


async def test_all(setup_test_project, glc: LoopCommunicator):  # pylint: disable=unused-argument, redefined-outer-name
    assert_image_count(0)
    assert GLOBALS.data_folder == '/tmp/learning_loop_lib_data'

    latest_model_id = await test_helper.get_latest_model_id()

    trainer = MockTrainer(model_format='mocked')
    context = Context(organization='zauberzeug', project='pytest')

    # TODO: maybe init call is missing

    training = Trainer.generate_training(context)
    training.model_id_for_detecting = latest_model_id
    trainer._training = training  # pylint: disable=protected-access
    await trainer._do_detections()  # pylint: disable=protected-access
    detections = trainer.active_training_io.det_load()

    assert_image_count(10)
    assert len(detections) == 10  # detections run on 10 images
    for img in detections:
        assert len(img['box_detections']) == 1
        assert len(img['point_detections']) == 1
        assert len(img['segmentation_detections']) == 1


def assert_image_count(value: int):
    images_folder = f'{GLOBALS.data_folder}/zauberzeug/pytest'
    files = glob(f'{images_folder}/**/*.*', recursive=True)
    files = [file for file in files if file.endswith('.jpg')]
    assert len(files) == value
