from typing import Dict, List, Optional, Union
from learning_loop_node.trainer import Trainer
from learning_loop_node.trainer.model import BasicModel, PretrainedModel
import subprocess
import asyncio
import logging
import time
from learning_loop_node.context import Context


class TestingTrainer(Trainer):
    __test__ = False

    def __init__(self, ) -> None:
        super().__init__('mocked')

    async def start_training(self) -> None:
        self.executor.start('while true; do sleep 1; done')

    @property
    def provided_pretrained_models(self) -> List[PretrainedModel]:
        return [
            PretrainedModel(name='small', label='Small', description='a small model'),
            PretrainedModel(name='medium', label='Medium', description='a medium model'),
            PretrainedModel(name='large', label='Large', description='a large model')]

    async def start_training(self, model: str = 'model.model') -> None:
        subprocess.call(['touch', '{self.training.training_folder}/{model}'])
        self.executor.start('while true; do sleep 1; done')

    async def start_training_from_scratch(self, id: str) -> None:
        await self.start_training(model=f'model_{id}.pt')

    def get_new_model(self) -> Optional[BasicModel]:
        return None

    async def _prepare(self) -> None:
        await super()._prepare()
        await asyncio.sleep(0.1)  # give tests a bit time to to check for the state

    async def _download_model(self) -> None:
        await super()._download_model()
        await asyncio.sleep(0.1)  # give tests a bit time to to check for the state

    async def upload_model(self) -> None:
        await asyncio.sleep(0.1)  # give tests a bit time to to check for the state
        result = await super().upload_model()
        await asyncio.sleep(0.1)  # give tests a bit time to to check for the state
        return result

    async def _upload_model(self, context: Context) -> dict:
        await asyncio.sleep(0.1)  # give tests a bit time to to check for the state
        result = await super()._upload_model(context)
        await asyncio.sleep(0.1)  # give tests a bit time to to check for the state
        return result

    def get_latest_model_files(self) -> Union[List[str], Dict[str, List[str]]]:
        time.sleep(1)  # NOTE reduce flakyness in Backend tests du to wrong order of events.
        fake_weight_file = '/tmp/weightfile.weights'
        with open(fake_weight_file, 'wb') as f:
            f.write(b'\x42')

        more_data_file = '/tmp/some_more_data.txt'
        with open(more_data_file, 'w') as f:
            f.write('zweiundvierzig')
        return {'mocked': [fake_weight_file, more_data_file], 'mocked_2': [fake_weight_file, more_data_file]}