from learning_loop_node.trainer.executor import Executor
from learning_loop_node.trainer.training_data import TrainingData
from learning_loop_node.context import Context
from learning_loop_node.trainer.training import Training
from mock_trainer import MockTrainer
import pytest
from learning_loop_node.trainer.model import Model
from uuid import uuid4
from learning_loop_node.globals import GLOBALS


def create_mock_trainer() -> MockTrainer:
    mock_trainer = MockTrainer(model_format='mocked')
    mock_trainer.executor = Executor(GLOBALS.data_folder)
    return mock_trainer


def test_get_model_files():
    mock_trainer = create_mock_trainer()
    files = mock_trainer.get_model_files('some_model_id')

    assert len(files) == 2
    assert 'weightfile.weights' in files[0]
    assert 'some_more_data.txt' in files[1]


@pytest.mark.asyncio
async def test_get_new_model():
    mock_trainer = create_mock_trainer()
    await mock_trainer.start_training()

    model = Model(id=(str(uuid4())))
    context = Context(organization="", project="")
    mock_trainer.training = Training(
        id=str(uuid4()),
        context=context,
        project_folder="",
        images_folder="")
    mock_trainer.training.data = TrainingData(image_data=[], categories=[])
    model = mock_trainer.get_new_model()
    assert model is not None
