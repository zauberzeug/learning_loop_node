from learning_loop_node.trainer.executor import Executor
from learning_loop_node.trainer.training_data import TrainingData
from learning_loop_node.context import Context
from learning_loop_node.trainer.training import Training
from mock_trainer import MockTrainer
import pytest
from typing import Generator
from learning_loop_node.tests import test_helper
from learning_loop_node.trainer.model import Model
from uuid import uuid4
from learning_loop_node import conftest


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


def create_mock_trainer() -> MockTrainer:
    mock_trainer = MockTrainer(model_format='mocked')
    mock_trainer.executor = Executor(conftest.data_folder_for_tests)
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
        base_model=model,
        context=context,
        project_folder="",
        images_folder="")
    mock_trainer.training.data = TrainingData(image_data=[], categories=[])
    model = mock_trainer.get_new_model()
    assert model is not None
