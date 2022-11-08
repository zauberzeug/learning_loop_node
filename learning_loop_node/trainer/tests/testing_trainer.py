from typing import List, Optional
from learning_loop_node.trainer import Trainer
from learning_loop_node.trainer.model import BasicModel, PretrainedModel
import subprocess
import asyncio
import logging


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


    async def _prepare(self) -> None:
        await super()._prepare()
        await asyncio.sleep(0.1)  # give tests a bit time to to check for the state

    async def _download_model(self) -> None:
        await super()._download_model()
        await asyncio.sleep(0.1)  # give tests a bit time to to check for the state
