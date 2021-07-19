#
from pydantic.main import BaseModel
from typing import List
import asyncio
from icecream import ic
import os
from glob import glob
from ..context import Context
from .. import node_helper
from ..loop import loop
from ..node_helper import download_images_data, create_resource_paths
from ..trainer.basic_data import BasicData
from ..trainer.training_data import TrainingData
import logging


class DataDownloader(BaseModel):
    context: Context
    data_query_params: str

    async def download_data(self, image_folder: str) -> TrainingData:
        basic_data = await self.download_basic_data()
        training_data = await self.download_images_and_annotations(basic_data, image_folder)
        return training_data

    async def download_basic_data(self) -> BasicData:
        async with loop.get(f'api/{self.context.organization}/projects/{self.context.project}/data?{self.data_query_params}') as response:
            assert response.status == 200, response
            basic_data = BasicData.parse_obj(await response.json())
            return basic_data

    async def download_images_and_annotations(self, basic_data: BasicData, image_folder) -> TrainingData:
        loop = asyncio.get_event_loop()
        image_data_coroutine = self.download_image_data(basic_data.image_ids)

        image_data_task = loop.create_task(image_data_coroutine)
        await self.download_images(loop, basic_data.image_ids, image_folder)

        image_data = await image_data_task
        logging.info(f'Done downloading image_data for {len(image_data)} images.')
        return TrainingData(image_data=image_data, box_categories=basic_data.box_categories)

    async def download_image_data(self, ids: List[str]) -> List[dict]:
        return await download_images_data(self.context.organization, self.context.project, ids)

    async def download_images(self, loop, image_ids: List[str], image_folder: str) -> None:
        paths, ids = create_resource_paths(self.context.organization, self.context.project,
                                           self.filter_needed_image_ids(image_ids, image_folder))
        await node_helper.download_images(loop, paths, ids, image_folder)

    @staticmethod
    def filter_needed_image_ids(all_image_ids, image_folder) -> List[str]:
        ids = [os.path.splitext(os.path.basename(image))[0] for image in glob(f'{image_folder}/*.jpg')]
        return [id for id in all_image_ids if id not in ids]
