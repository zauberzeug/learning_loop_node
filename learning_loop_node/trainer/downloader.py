import logging
import os
from typing import Dict, List, Tuple

from ..data_exchanger import DataExchanger


class TrainingsDownloader():

    def __init__(self, data_exchanger: DataExchanger, training_number: int):
        self.data_exchanger = data_exchanger
        self.training_number = training_number

    async def download_training_data(self, image_folder: str) -> Tuple[List[Dict], int]:
        """Download the images the loop chose for the training, with their annotations."""
        image_ids = await self.data_exchanger.fetch_training_image_uuids(self.training_number)
        image_data, skipped_image_count = await self.download_images_and_annotations(image_ids, image_folder)
        return (image_data, skipped_image_count)

    async def download_images_and_annotations(self, image_ids: List[str], image_folder: str) -> Tuple[List[Dict], int]:
        image_data = await self.data_exchanger.download_images_data(image_ids)
        await self.data_exchanger.download_images(image_ids, image_folder)
        logging.info('filtering corrupt images')  # download only safes valid images
        valid_image_data: List[Dict] = []
        skipped_image_count = 0
        for i in image_data:
            if os.path.isfile(f'{image_folder}/{i["id"]}.jpg'):
                valid_image_data.append(i)
            else:
                skipped_image_count += 1
        logging.info('Done downloading image data for %s images.', len(image_data))
        return (valid_image_data, skipped_image_count)
