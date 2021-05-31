
from typing import List
from learning_loop_node.converter.converter import Converter
import asyncio


class MockConverter(Converter):

    async def _convert(self) -> None:
        await asyncio.sleep(1)

    def get_converted_files(self, model_id: str) -> List[str]:
        fake_converted_file = '/tmp/converted_weightfile.converted'
        with open(fake_converted_file, 'wb') as f:
            f.write(b'\x42')
        return [fake_converted_file]
