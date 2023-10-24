import logging
from typing import List

import pytest

from learning_loop_node.converter.converter_logic import ConverterLogic
from learning_loop_node.converter.converter_node import ConverterNode
from learning_loop_node.data_classes import ModelInformation
from learning_loop_node.loop_communication import LoopCommunicator
from learning_loop_node.tests import test_helper


class TestConverter(ConverterLogic):
    __test__ = False  # hint for pytest

    def __init__(self, source_format: str, target_format: str,  models: List[ModelInformation]):
        super().__init__(source_format, target_format)
        self.models = models

    async def _convert(self, model_information: ModelInformation) -> None:
        self.models.append(model_information)

    def get_converted_files(self, model_id) -> List[str]:
        return []  # test: test_meta_information fails because model cannot be uploaded


@pytest.mark.asyncio
@pytest.fixture()
async def setup_converter_test_project(glc: LoopCommunicator):
    await glc.delete("/zauberzeug/projects/pytest_conv?keep_images=true")
    project_configuration = {
        'project_name': 'pytest_conv', 'box_categories': 1, 'point_categories': 1, 'inbox': 0, 'annotate': 0, 'review': 0,
        'complete': 0, 'image_style': 'plain', 'thumbs': False, 'trainings': 1}
    r = await glc.post("/zauberzeug/projects/generator", json=project_configuration)
    assert r.status_code == 200
    yield
    await glc.delete("/zauberzeug/projects/pytest?keep_images=true")


# pylint: disable=redefined-outer-name, unused-argument
@pytest.mark.asyncio
async def test_meta_information(setup_converter_test_project):
    model_id = await test_helper.get_latest_model_id(project='pytest_conv')

    converter = TestConverter(source_format='mocked', target_format='test', models=[])
    node = ConverterNode(name='test', converter=converter)
    await node.convert_models()

    pytest_project_model = [m for m in converter.models if m.id == model_id][0]

    categories = pytest_project_model.categories
    assert len(categories) == 2
    category_types = [category.type for category in categories]
    assert 'box' in category_types
    assert 'point' in category_types
