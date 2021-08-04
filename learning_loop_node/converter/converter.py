from ..converter.model_information import ModelInformation
from typing import List, Optional
import os
from pydantic.main import BaseModel
from learning_loop_node.node import Node
from icecream import ic
from learning_loop_node import node_helper
import shutil


class Converter(BaseModel):
    model_folder: Optional[str]
    source_format: str
    target_format: str

    async def convert(self, model_information: ModelInformation) -> None:
        project_folder = Node.create_project_folder(model_information.context)

        self.model_folder = Converter.create_model_folder(project_folder, model_information.model_id)
        await node_helper.download_model(self.model_folder, model_information.context, model_information. model_id, self.source_format)

        await self._convert(model_information)

    async def _convert(self, model_information: ModelInformation) -> None:
        raise NotImplementedError()

    def get_converted_files(self, model_id) -> List[str]:
        raise NotImplementedError()

    async def upload_model(self, context, model_id: str) -> bool:
        files = self.get_converted_files(model_id)
        await node_helper.upload_model(context, files, model_id, self.target_format)

    @staticmethod
    def create_convert_folder(project_folder: str) -> str:
        image_folder = f'{project_folder}/images'
        os.makedirs(image_folder, exist_ok=True)
        return image_folder

    @staticmethod
    def create_model_folder(project_folder: str, model_id: str) -> str:
        model_folder = f'{project_folder}/{model_id}'
        shutil.rmtree(model_folder, ignore_errors=True)  # cleanup
        os.makedirs(model_folder, exist_ok=True)
        return model_folder
