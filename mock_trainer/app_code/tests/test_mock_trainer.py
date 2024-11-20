from typing import Dict
from uuid import uuid4

import pytest

from learning_loop_node.data_classes import Context, TrainerState, Training
from learning_loop_node.globals import GLOBALS
from learning_loop_node.trainer.executor import Executor

from ..mock_trainer_logic import MockTrainerLogic

# pylint: disable=protected-access
# pylint: disable=unused-argument


async def create_mock_trainer() -> MockTrainerLogic:
    mock_trainer = MockTrainerLogic(model_format='mocked')
    mock_trainer._executor = Executor(GLOBALS.data_folder)
    return mock_trainer


@pytest.mark.usefixtures('setup_test_project2')
async def test_get_model_files():
    mock_trainer = await create_mock_trainer()
    files = await mock_trainer._get_latest_model_files()

    assert isinstance(files, Dict)

    assert len(files) == 2
    assert files['mocked'] == ['/tmp/weightfile.weights', '/tmp/some_more_data.txt']
    assert files['mocked_2'] == ['/tmp/weightfile.weights', '/tmp/some_more_data.txt']


@pytest.mark.usefixtures('setup_test_project2')
async def test_get_new_model():
    mock_trainer = await create_mock_trainer()
    await mock_trainer._start_training_from_base_model()

    context = Context(organization="", project="")
    mock_trainer._training = Training(  # pylint: disable=protected-access
        id=str(uuid4()),
        context=context,
        project_folder="",
        images_folder="",
        training_folder="",
        categories=[],
        hyperparameters={},
        model_variant='',
        training_number=0,
        training_state=TrainerState.Preparing)
    mock_trainer._training.image_data = []
    model = mock_trainer._get_new_best_training_state()
    assert model is not None
