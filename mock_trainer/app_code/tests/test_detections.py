from dataclasses import asdict
from glob import glob

from fastapi.encoders import jsonable_encoder

from learning_loop_node.data_classes import Category, Context
from learning_loop_node.globals import GLOBALS
from learning_loop_node.helpers.misc import create_project_folder, generate_training
from learning_loop_node.loop_communication import LoopCommunicator
from learning_loop_node.tests import test_helper
from learning_loop_node.trainer.trainer_node import TrainerNode

from ..mock_trainer_logic import MockTrainerLogic

# pylint: disable=protected-access,redefined-outer-name,unused-argument


async def test_all(setup_test_project1, glc: LoopCommunicator):
    assert_image_count(0)
    assert GLOBALS.data_folder == '/tmp/learning_loop_lib_data'

    latest_model_id = await test_helper.get_latest_model_id(project='pytest_mock_trainer_test1')

    trainer = MockTrainerLogic(model_format='mocked')
    node = TrainerNode(name='test', trainer_logic=trainer)
    context = Context(organization='zauberzeug', project='pytest_mock_trainer_test1')
    details = {'categories': [jsonable_encoder(asdict(Category(id='some_id', name='some_category_name')))],
               'id': '798bfbf1-8948-2ea9-32fb-3571b6748bca',  # version 1.2 of demo project
               'training_number': 0,
               'resolution': 800,
               'flip_rl': False,
               'flip_ud': False}
    # await asyncio.sleep(100)

    trainer._node = node
    trainer._init_new_training(context=context, details=details)

    project_folder = create_project_folder(context)
    training = generate_training(project_folder, context)
    training.model_uuid_for_detecting = latest_model_id
    trainer._training = training
    await trainer._do_detections()
    detections = trainer.active_training_io.load_detections()

    assert_image_count(10)  # TODO This assert fails frequently on Drone
    assert len(detections) == 10  # detections run on 10 images
    for img in detections:
        assert len(img.box_detections) == 1
        assert len(img.point_detections) == 1
        assert len(img.segmentation_detections) == 1


def assert_image_count(value: int):
    images_folder = f'{GLOBALS.data_folder}/zauberzeug/pytest_mock_trainer_test1'
    files = glob(f'{images_folder}/**/*.*', recursive=True)
    files = [file for file in files if file.endswith('.jpg')]
    assert len(files) == value
