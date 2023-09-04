import logging
import os
from typing import Dict, List, Optional, Tuple

from learning_loop_node.data_classes import Context
from learning_loop_node.rest_helpers.downloader import DataDownloader


class TrainingsDownloader():
    downloader: DataDownloader
    data_query_params: Optional[str]

    def __init__(self, context: Context, data_query_params: Optional[str] = 'state=complete'):
        self.data_query_params = data_query_params
        self.downloader = DataDownloader(context)

    async def download_training_data(self, image_folder: str) -> Tuple[List[Dict], int]:
        image_ids = await self.downloader.fetch_image_ids(query_params=self.data_query_params)
        image_data, skipped_image_count = await self.download_images_and_annotations(image_ids, image_folder)
        return (image_data, skipped_image_count)

    async def download_images_and_annotations(self, image_ids: List[str], image_folder: str) -> Tuple[List[Dict], int]:
        await self.downloader.download_images(image_ids, image_folder)
        image_data = await self.downloader.download_images_data(image_ids)
        logging.info('filtering corrupt images')  # download only safes valid images
        valid_image_data: List[Dict] = []
        skipped_image_count = 0
        for i in image_data:
            if os.path.isfile(f'{image_folder}/{i["id"]}.jpg'):
                valid_image_data.append(i)
            else:
                skipped_image_count += 1
        logging.info(f'Done downloading image data for {len(image_data)} images.')
        return (valid_image_data, skipped_image_count)
