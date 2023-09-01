
import asyncio
from typing import List

from learning_loop_node.converter.converter_model import ConverterModel
from learning_loop_node.data_classes import ModelInformation


class MockConverterModel(ConverterModel):

    async def _convert(self, model_information: ModelInformation) -> None:
        await asyncio.sleep(1)

    def get_converted_files(self, model_id: str) -> List[str]:
        fake_converted_file = '/tmp/converted_weightfile.converted'
        with open(fake_converted_file, 'wb') as f:
            f.write(b'\x42')
        return [fake_converted_file]
