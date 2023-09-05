import json
import os
import shutil
from abc import abstractmethod
from typing import List, Optional

from ..data_classes import ModelInformation
from ..node import Node


class ConverterLogic():

    def __init__(
            self, source_format: str, target_format: str):
        self.source_format = source_format
        self.target_format = target_format
        self._node: Optional[Node] = None
        self.model_folder: Optional[str] = None

    def init(self, node: Node) -> None:
        self._node = node

    @property
    def node(self) -> Node:
        if self._node is None:
            raise Exception('ConverterLogic not initialized')
        return self._node

    async def convert(self, model_information: ModelInformation) -> None:
        project_folder = Node.create_project_folder(model_information.context)

        self.model_folder = ConverterLogic.create_model_folder(project_folder, model_information.id)
        await self.node.data_exchanger.download_model(self.model_folder,
                                                      model_information.context,
                                                      model_information.id,
                                                      self.source_format)

        with open(f'{self.model_folder}/model.json', 'r') as f:
            content = json.load(f)
            if 'resolution' in content:
                model_information.resolution = content['resolution']

        await self._convert(model_information)

    async def upload_model(self, context, model_id: str) -> None:
        files = self.get_converted_files(model_id)
        await self.node.data_exchanger.upload_model(context, files, model_id, self.target_format)

    @abstractmethod
    async def _convert(self, model_information: ModelInformation) -> None:
        """Converts the model in self.model_folder to the target format."""

    @abstractmethod
    def get_converted_files(self, model_id) -> List[str]:
        """Returns a list of files that should be uploaded to the server."""

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
