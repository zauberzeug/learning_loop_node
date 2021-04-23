import pytest
from typing import Generator
from requests import Session
from urllib.parse import urljoin
import mock_trainer_tests.test_helper as test_helper
import main
import asyncio
from learning_loop_node.trainer.model import Model


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
async def test_start_step_stop(web: Session):
    def get_datapoints():
        response = web.get(f'/api/zauberzeug/projects/pytest/trainings')
        assert response.status_code == 200
        return response.json()['charts'][0]['data']

    await main.trainer.connect()
    await main.trainer.sio.sleep(1.0)

    datapoints = get_datapoints()
    assert len(datapoints) == 11
    model = datapoints[0]

    source_model = Model(id=model['model_id'])
    begin_training_handler = main.trainer.sio.handlers['/']['begin_training']
    await begin_training_handler('zauberzeug', 'pytest', source_model=source_model)

    await assert_training_started()
    step_result = await main._step()
    assert step_result is not None

    datapoints = get_datapoints()
    assert len(datapoints) == 12


async def assert_training_started():
    await asyncio.sleep(0.1)
    assert main.trainer.training.data != None
