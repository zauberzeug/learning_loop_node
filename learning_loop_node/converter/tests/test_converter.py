from typing import List
import pytest

import logging
from learning_loop_node.model_information import ModelInformation
from learning_loop_node.converter.converter import Converter
from learning_loop_node.converter.converter_node import ConverterNode


class MockedConverter(Converter):
    models: List[ModelInformation] = []

    async def _convert(self, model_information: ModelInformation) -> None:
        self.models.append(model_information)


@pytest.mark.asyncio
async def test_meta_informations():
    converter = MockedConverter(source_format='mocked', target_format='test')
    node = ConverterNode(name='test', converter=converter)
    await node.convert_models()
    assert len(converter.models) == 3
    categories = converter.models[0].categories
    assert len(categories) == 6
    assert categories[0].type == 'box'
    assert categories[1].type == 'point'
