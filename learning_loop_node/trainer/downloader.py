import logging
import os
from collections.abc import Collection, Mapping
from typing import Dict, List, Optional, Tuple

from ..data_exchanger import DataExchanger
from .subsets import restrict_to_subsets


class TrainingsDownloader():

    def __init__(self, data_exchanger: DataExchanger, data_query_params: Optional[str] = 'state=complete',
                 subsets: Mapping[str, Collection[str]] | None = None):
        """
        :param subsets: `{set: [subset names]}` of the restricted sets.
        """
        self.data_query_params = data_query_params
        self.data_exchanger = data_exchanger
        self.subsets = subsets or {}

    async def download_training_data(self, image_folder: str) -> Tuple[List[Dict], int]:
        image_ids = await self.data_exchanger.fetch_image_uuids(query_params=self.data_query_params)
        image_data, skipped_image_count = await self.download_images_and_annotations(image_ids, image_folder)
        return (image_data, skipped_image_count)

    async def download_images_and_annotations(self, image_ids: List[str], image_folder: str) -> Tuple[List[Dict], int]:
        image_data = await self.data_exchanger.download_images_data(image_ids)
        if self.subsets:
            image_data = restrict_to_subsets(image_data, self.subsets)
            image_ids = [image['id'] for image in image_data]
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
