from typing import List, Optional, Tuple
from learning_loop_node.context import Context
from learning_loop_node.rest.downloader import DataDownloader
from learning_loop_node.task_logger import create_task
from learning_loop_node.trainer.training_data import TrainingData
import logging
import os
from icecream import ic


class TrainingsDownloader():
    downloader: DataDownloader
    data_query_params: str

    def __init__(self, context: Context, data_query_params: Optional[str] = 'state=complete'):
        self.data_query_params = data_query_params
        self.downloader = DataDownloader(context)

    async def download_training_data(self, image_folder: str) -> Tuple[List[dict], int]:
        image_ids = await self.downloader.fetch_image_ids(query_params=self.data_query_params)
        image_data, skipped_image_count = await self.download_images_and_annotations(image_ids, image_folder)
        return (image_data, skipped_image_count)

    async def download_images_and_annotations(self, image_ids: List[str], image_folder: str) -> TrainingData:
        await self.downloader.download_images(image_ids, image_folder)
        image_data = await self.downloader.download_images_data(image_ids)
        logging.info('filtering corrupt images')  # download only safes valid images
        valid_image_data: List[dict] = []
        skipped_image_count = 0
        for i in image_data:
            if os.path.isfile(f'{image_folder}/{i["id"]}.jpg'):
                valid_image_data.append(i)
            else:
                skipped_image_count += 1
        logging.info(f'Done downloading image data for {len(image_data)} images.')
        return (valid_image_data, skipped_image_count)
