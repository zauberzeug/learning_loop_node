from typing import List, Optional
import os
from glob import glob
import asyncio
from learning_loop_node.context import Context
from learning_loop_node.loop import loop
from learning_loop_node.rest import downloads
from learning_loop_node import node_helper
import logging
from icecream import ic
from time import perf_counter


class DataDownloader():
    context: Context

    def __init__(self, context: Context):
        self.context = context

    async def fetch_image_ids(self, query_params: Optional[str] = '') -> List[str]:
        async with loop.get(f'api/{self.context.organization}/projects/{self.context.project}/data?{query_params}') as response:
            assert response.status == 200, response
            return (await response.json())['image_ids']

    async def download_images_data(self, ids: List[str]) -> List[dict]:
        return await downloads.download_images_data(self.context.organization, self.context.project, ids)

    async def download_images(self, image_ids: List[str], image_folder: str) -> None:
        '''Will skip existing images'''
        new_image_ids = await asyncio.get_event_loop().run_in_executor(None, DataDownloader.filter_existing_images, image_ids, image_folder)
        paths, ids = node_helper.create_resource_paths(self.context.organization, self.context.project, new_image_ids)
        await downloads.download_images(paths, ids, image_folder)

    @staticmethod
    def filter_existing_images(all_image_ids, image_folder) -> List[str]:
        logging.info(f'### Going to filter {len(all_image_ids)} images ids')
        start = perf_counter()
        ids = [os.path.splitext(os.path.basename(image))[0]
               for image in glob(f'{image_folder}/*.jpg')]
        logging.info(f'found {len(ids)} images on disc')
        result = [id for id in all_image_ids if id not in ids]
        end = perf_counter()
        logging.info(f'calculated {len(result)} new image ids, which took {end-start:0.2f} seconds')
        return result
