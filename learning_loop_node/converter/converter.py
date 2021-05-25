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
    downloader: Optional[Downloader]

    async def convert(self, context: Context, source_model: dict, downloader: Downloader) -> None:
        self.downloader = downloader
        project_folder = Node.create_project_folder(context.organization, context.project)

        self.model_folder = Converter.create_model_folder(project_folder, source_model['id'])
        self.downloader.download_model(self.model_folder, source_model['id'])

        await self._convert()

    async def _convert(self) -> None:
        raise NotImplementedError()

    def is_conversion_alive(self) -> bool:
        raise NotImplementedError()

    def get_converted_files(self, model_id) -> List[str]:
        raise NotImplementedError()

    async def save_model(self, host_url, headers, organization, project, model_id: str) -> bool:
        # TODO remove the need of host_url. Create an uploader?
        files = self.get_converted_files(model_id)

        uri_base = f'{host_url}/api/{organization}/projects/{project}'
        data = aiohttp.FormData()

        for file_name in files:
            data.add_field('files',  open(file_name, 'rb'))

        async with aiohttp.ClientSession() as session:
            async with session.put(f'{uri_base}/models/{model_id}/file', data=data, headers=headers) as response:
                if response.status != 200:
                    msg = f'---- could not save model with id {model_id}'
                    raise Exception(msg)
                else:
                    ic(f'---- uploaded model with id {model_id}')

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
