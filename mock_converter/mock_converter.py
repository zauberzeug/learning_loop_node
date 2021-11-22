
from typing import List
from learning_loop_node.converter.converter import Converter
import asyncio
from learning_loop_node.model_information import ModelInformation


class MockConverter(Converter):

    async def _convert(self, model_information: ModelInformation) -> None:
        await asyncio.sleep(1)

    def get_converted_files(self, model_id: str) -> List[str]:
        fake_converted_file = '/tmp/converted_weightfile.converted'
        with open(fake_converted_file, 'wb') as f:
            f.write(b'\x42')
        return [fake_converted_file]
