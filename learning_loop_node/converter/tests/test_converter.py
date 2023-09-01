from typing import List

import pytest

from learning_loop_node.converter.converter_model import ConverterModel
from learning_loop_node.converter.converter_node import ConverterNode
from learning_loop_node.data_classes import ModelInformation
from learning_loop_node.tests import test_helper


class MockedConverter(ConverterModel):
    models: List[ModelInformation] = []

    async def _convert(self, model_information: ModelInformation) -> None:
        self.models.append(model_information)

    def get_converted_files(self, model_id) -> List[str]:
        return []


@pytest.fixture()
def setup_converter_test_project():
    test_helper.LiveServerSession().delete("/zauberzeug/projects/pytest?keep_images=true")
    project_configuration = {
        'project_name': 'pytest', 'box_categories': 1, 'point_categories': 1, 'inbox': 0, 'annotate': 0, 'review': 0,
        'complete': 0, 'image_style': 'plain', 'thumbs': False, 'trainings': 1}
    assert test_helper.LiveServerSession().post("/zauberzeug/projects/generator",
                                                json=project_configuration).status_code == 200
    yield
    test_helper.LiveServerSession().delete("/zauberzeug/projects/pytest?keep_images=true")


# pylint: disable=unused-argument
async def test_meta_information(setup_converter_test_project):
    model_id = await test_helper.get_latest_model_id()

    converter = MockedConverter(source_format='mocked', target_format='test')
    node = ConverterNode(name='test', converter=converter)
    await node.convert_models()

    pytest_project_model = [m for m in converter.models if m.id == model_id][0]

    categories = pytest_project_model.categories
    assert len(categories) == 2
    category_types = [category.ctype for category in categories]
    assert 'box' in category_types
    assert 'point' in category_types
