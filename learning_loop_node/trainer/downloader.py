from typing import Optional
from learning_loop_node.context import Context
from learning_loop_node.rest.downloader import DataDownloader
from learning_loop_node.task_logger import create_task
from learning_loop_node.trainer.training_data import TrainingData
from learning_loop_node.data_classes.basic_data import BasicData
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
        basic_data = await self.downloader.download_basic_data(query_params=self.data_query_params)
        training_data = await self.download_images_and_annotations(basic_data, image_folder)
        return training_data

    async def download_images_and_annotations(self, basic_data: BasicData, image_folder: str) -> TrainingData:
        await self.downloader.download_images(basic_data.image_ids, image_folder)
        image_data = await self.downloader.download_images_data(basic_data.image_ids)
        logging.info('filtering corrupt images')  # download only safes valid images
        training_data = TrainingData(image_data=[], categories={c['name']: c['id'] for c in basic_data.categories})
        for i in image_data:
            if os.path.isfile(f'{image_folder}/{i["id"]}.jpg'):
                training_data.image_data.append(i)
            else:
                training_data.skipped_image_count += 1
        logging.info(f'Done downloading image data for {len(image_data)} images.')
        return training_data
