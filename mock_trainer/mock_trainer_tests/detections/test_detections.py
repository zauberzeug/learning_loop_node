import logging
from learning_loop_node.context import Context
from learning_loop_node.data_classes import Category
from mock_trainer import MockTrainer
import pytest
from learning_loop_node.globals import GLOBALS
from mock_trainer import MockTrainer
from learning_loop_node.tests import test_helper
from glob import glob
from icecream import ic


@pytest.fixture()
def create_project():
    test_helper.LiveServerSession().delete(f"/api/zauberzeug/projects/pytest?keep_images=true")
    project_configuration = {'project_name': 'pytest', 'inbox': 1, 'annotate': 2, 'review': 3, 'complete': 4, 'image_style': 'plain',
                             'box_categories': 1, 'point_categories': 1, 'segmentation_categories': 1, 'thumbs': False}
    assert test_helper.LiveServerSession().post(f"/api/zauberzeug/projects/generator",
                                                json=project_configuration).status_code == 200
    yield
    test_helper.LiveServerSession().delete(f"/api/zauberzeug/projects/pytest?keep_images=true")


@pytest.mark.asyncio
async def test_all(create_project):
    assert_image_count(0)
    assert GLOBALS.data_folder == '/tmp/learning_loop_lib_data'

    model_id = await test_helper.assert_upload_model()
    trainer = MockTrainer(model_format='mocked')
    context = Context(organization='zauberzeug', project='pytest')

    detections = await trainer.do_detections(context=context, model_id=model_id, model_format='mocked')
    assert_image_count(10)
    assert len(detections) == 10  # detections run on 10 images
    for img in detections:
        assert len(img['box_detections']) == 1
        assert len(img['point_detections']) == 1
        assert len(img['segmentation_detections']) == 1


def assert_image_count(value: int):
    images_folder = f'{GLOBALS.data_folder}/zauberzeug/pytest'
    files = glob(f'{images_folder}/**/*.*', recursive=True)
    assert len(files) == value
