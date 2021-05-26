from learning_loop_node.synchronizer.model_syncronizer import ModelSynchronizer
import aiohttp
from typing import List, Optional
import os
from pydantic.main import BaseModel
from learning_loop_node.trainer.downloader import Downloader
from learning_loop_node.context import Context
from learning_loop_node.node import Node
from icecream import ic


class Converter(BaseModel):
    model_folder: Optional[str]

    async def convert(self, context: Context, source_model: dict, model_synchronizer: ModelSynchronizer) -> None:
        project_folder = Node.create_project_folder(context.organization, context.project)

        self.model_folder = Converter.create_model_folder(project_folder, source_model['id'])
        model_synchronizer.download(self.model_folder, source_model['id'])

        await self._convert()

    async def _convert(self) -> None:
        raise NotImplementedError()

    def is_conversion_alive(self) -> bool:
        raise NotImplementedError()

    def get_converted_files(self, model_id) -> List[str]:
        raise NotImplementedError()

    async def save_model(self, model_id: str, model_synchronizer: ModelSynchronizer) -> bool:
        files = self.get_converted_files(model_id)
        await model_synchronizer.upload(files, model_id)

    @staticmethod
    def create_convert_folder(project_folder: str) -> str:
        image_folder = f'{project_folder}/images'
        os.makedirs(image_folder, exist_ok=True)
        return image_folder

    @staticmethod
    def create_model_folder(project_folder: str, model_id: str) -> str:
        model_folder = f'{project_folder}/{model_id}'
        os.makedirs(model_folder, exist_ok=True)
        return model_folder
