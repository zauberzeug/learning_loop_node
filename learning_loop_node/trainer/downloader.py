from learning_loop_node.context import Context
from pydantic.main import BaseModel
from typing import List
from learning_loop_node.trainer.training_data import TrainingData
import requests
import asyncio
from learning_loop_node.node_helper import download_images_data, create_resource_urls
from learning_loop_node import node_helper
from icecream import ic
import os
from glob import glob
from learning_loop_node.trainer.basic_data import BasicData


class DataDownloader(BaseModel):
    server_base_url: str
    headers: dict
    context: Context
    data_query_params: str

    def get_data_url(self) -> str:
        return f'{self.server_base_url}/api/{self.context.organization}/projects/{self.context.project}/data?{self.data_query_params}'

    async def download_data(self, image_folder: str) -> TrainingData:
        basic_data = self.download_basic_data()
        training_data = await self.download_images_and_annotations(basic_data, image_folder)
        return training_data

    def download_basic_data(self) -> BasicData:
        response = requests.get(self.get_data_url(), headers=self.headers)
        assert response.status_code == 200
        basic_data = BasicData.parse_obj(response.json())
        return basic_data

    async def download_images_and_annotations(self, basic_data: BasicData, image_folder) -> TrainingData:
        loop = asyncio.get_event_loop()
        image_data_coroutine = self.download_image_data(basic_data.image_ids)

        image_data_task = loop.create_task(image_data_coroutine)
        await self.download_images(loop, basic_data.image_ids, image_folder)

        image_data = await image_data_task
        ic(f'Done downloading image_data for {len(image_data)} images.')
        return TrainingData(image_data=image_data, box_categories=basic_data.box_categories)

    async def download_image_data(self, ids: List[str]) -> List[dict]:
        return await download_images_data(
            self.server_base_url, self.headers, self.context.organization, self.context.project, ids)

    async def download_images(self, loop, image_ids: List[str], image_folder: str) -> None:
        urls, ids = create_resource_urls(
            self.server_base_url, self.context.organization, self.context.project, self.filter_needed_image_ids(image_ids, image_folder))
        await node_helper.download_images(loop, urls, ids, self.headers, image_folder)

    @staticmethod
    def filter_needed_image_ids(all_image_ids, image_folder) -> List[str]:
        ids = [os.path.splitext(os.path.basename(image))[0] for image in glob(f'{image_folder}/*.jpg')]
        return [id for id in all_image_ids if id not in ids]
