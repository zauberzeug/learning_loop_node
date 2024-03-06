from dataclasses import asdict
from glob import glob

from fastapi.encoders import jsonable_encoder

from learning_loop_node.data_classes import Category, Context
from learning_loop_node.globals import GLOBALS
from learning_loop_node.helpers.misc import create_project_folder
from learning_loop_node.loop_communication import LoopCommunicator
from learning_loop_node.tests import test_helper
from learning_loop_node.trainer.trainer_logic import TrainerLogic
from learning_loop_node.trainer.trainer_node import TrainerNode

from ..mock_trainer_logic import MockTrainerLogic


async def test_all(setup_test_project1, glc: LoopCommunicator):  # pylint: disable=unused-argument, redefined-outer-name
    assert_image_count(0)
    assert GLOBALS.data_folder == '/tmp/learning_loop_lib_data'

    latest_model_id = await test_helper.get_latest_model_id(project='pytest_p1')

    trainer = MockTrainerLogic(model_format='mocked')
    node = TrainerNode(name='test', trainer_logic=trainer)
    context = Context(organization='zauberzeug', project='pytest_p1')
    details = {'categories': [jsonable_encoder(asdict(Category(id='some_id', name='some_category_name')))],
               'id': '917d5c7f-403d-7e92-f95f-577f79c2273a',  # version 1.2 of demo project
               'training_number': 0,
               'resolution': 800,
               'flip_rl': False,
               'flip_ud': False}
    trainer._node = node  # pylint: disable=protected-access
    trainer.init_new_training(context=context, details=details)

    project_folder = create_project_folder(context)
    training = TrainerLogic.generate_training(project_folder, context)
    training.model_id_for_detecting = latest_model_id
    trainer._training = training  # pylint: disable=protected-access
    await trainer._do_detections()  # pylint: disable=protected-access
    detections = trainer.active_training_io.load_detections()

    assert_image_count(10)  # TODO This assert fails frequently on Drone
    assert len(detections) == 10  # detections run on 10 images
    for img in detections:
        assert len(img.box_detections) == 1
        assert len(img.point_detections) == 1
        assert len(img.segmentation_detections) == 1


def assert_image_count(value: int):
    images_folder = f'{GLOBALS.data_folder}/zauberzeug/pytest_p1'
    files = glob(f'{images_folder}/**/*.*', recursive=True)
    files = [file for file in files if file.endswith('.jpg')]
    assert len(files) == value
