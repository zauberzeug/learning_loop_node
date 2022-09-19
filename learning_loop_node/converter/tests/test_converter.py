from typing import List
import pytest

import logging
from learning_loop_node.model_information import ModelInformation
from learning_loop_node.converter.converter import Converter
from learning_loop_node.converter.converter_node import ConverterNode
from learning_loop_node.tests import test_helper


class MockedConverter(Converter):
    models: List[ModelInformation] = []

    async def _convert(self, model_information: ModelInformation) -> None:
        self.models.append(model_information)


@pytest.fixture()
def create_project():
    test_helper.LiveServerSession().delete(f"/api/zauberzeug/projects/pytest?keep_images=true")
    project_configuration = {'project_name': 'pytest', 'box_categories': 1,  'point_categories': 1, 'inbox': 0, 'annotate': 0, 'review': 0, 'complete': 0, 'image_style': 'plain',
                             'thumbs': False, 'trainings': 1}
    assert test_helper.LiveServerSession().post(f"/api/zauberzeug/projects/generator",
                                                json=project_configuration).status_code == 200
    yield
    test_helper.LiveServerSession().delete(f"/api/zauberzeug/projects/pytest?keep_images=true")


async def test_meta_information(create_project):
    model_id = await test_helper.get_latest_model_id()

    converter = MockedConverter(source_format='mocked', target_format='test')
    node = ConverterNode(name='test', converter=converter)
    await node.convert_models()

    pytest_project_model = [m for m in converter.models if m.id == model_id][0]

    categories = pytest_project_model.categories
    assert len(categories) == 2
    assert categories[0].type == 'box'
    assert categories[1].type == 'point'
