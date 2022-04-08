from typing import List, Optional
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

    async def download_training_data(self, image_folder: str) -> TrainingData:
        image_ids = await self.downloader.fetch_image_ids(query_params=self.data_query_params)
        training_data = await self.download_images_and_annotations(image_ids, image_folder)
        return training_data

    async def download_images_and_annotations(self, image_ids: List[str], image_folder: str) -> TrainingData:
        await self.downloader.download_images(image_ids, image_folder)
        image_data = await self.downloader.download_images_data(image_ids)
        logging.info('filtering corrupt images')  # download only safes valid images
        training_data = TrainingData()
        for i in image_data:
            if os.path.isfile(f'{image_folder}/{i["id"]}.jpg'):
                training_data.image_data.append(i)
            else:
                training_data.skipped_image_count += 1
        logging.info(f'Done downloading image data for {len(image_data)} images.')
        return training_data
