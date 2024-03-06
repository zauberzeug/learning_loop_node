import logging
import os
from typing import Dict, List, Optional, Tuple

from ..data_exchanger import DataExchanger


class TrainingsDownloader():

    def __init__(self, data_exchanger: DataExchanger, data_query_params: Optional[str] = 'state=complete'):
        self.data_query_params = data_query_params
        self.data_exchanger = data_exchanger

    async def download_training_data(self, image_folder: str) -> Tuple[List[Dict], int]:
        image_ids = await self.data_exchanger.fetch_image_uuids(query_params=self.data_query_params)
        image_data, skipped_image_count = await self.download_images_and_annotations(image_ids, image_folder)
        return (image_data, skipped_image_count)

    async def download_images_and_annotations(self, image_ids: List[str], image_folder: str) -> Tuple[List[Dict], int]:
        await self.data_exchanger.download_images(image_ids, image_folder)
        image_data = await self.data_exchanger.download_images_data(image_ids)
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
