from typing import List, Optional
import shutil
import os
from glob import glob
import asyncio
from learning_loop_node.context import Context
from learning_loop_node.data_classes.basic_data import BasicData
from learning_loop_node.loop import loop
from learning_loop_node.rest import downloads
from learning_loop_node import node_helper
import logging
from icecream import ic


class DataDownloader():
    context: Context

    def __init__(self, context: Context):
        self.context = context

        self.check_jpeg = shutil.which('jpeginfo') is not None
        if self.check_jpeg:
            logging.info(
                'Detected command line tool "jpeginfo". Images will be checked for validity')
        else:
            logging.error(
                'Missing command line tool "jpeginfo". We can not check for validity of images.')

    async def download_basic_data(self, query_params: Optional[str] = '') -> BasicData:
        async with loop.get(f'api/{self.context.organization}/projects/{self.context.project}/data?{query_params}') as response:
            assert response.status == 200, response
            basic_data = BasicData.parse_obj(await response.json())
            return basic_data

    async def download_images_data(self, ids: List[str]) -> List[dict]:
        return await downloads.download_images_data(self.context.organization, self.context.project, ids)

    async def download_images(self, image_ids: List[str], image_folder: str) -> None:
        paths, ids = node_helper.create_resource_paths(self.context.organization, self.context.project,
                                                       DataDownloader.filter_existing_images(image_ids, image_folder))  # hier
        await downloads.download_images(paths, ids, image_folder)

    async def is_valid_image(self, file):
        if not os.path.isfile(file):
            return False
        if not self.check_jpeg:
            return True

        info = await asyncio.create_subprocess_shell(
            f'jpeginfo -c {file}',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        out, err = await info.communicate()
        return "[OK]" in out.decode()

    @staticmethod
    def filter_existing_images(all_image_ids, image_folder) -> List[str]:
        ids = [os.path.splitext(os.path.basename(image))[0]
               for image in glob(f'{image_folder}/*.jpg')]
        return [id for id in all_image_ids if id not in ids]
