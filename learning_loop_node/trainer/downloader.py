import logging
import os
from collections.abc import Collection, Mapping
from typing import Dict, List, Optional, Tuple

from ..data_exchanger import DataExchanger
from .states import restrict_to_states, states_to_fetch
from .subsets import restrict_to_subsets


class TrainingsDownloader():

    def __init__(self, data_exchanger: DataExchanger, data_query_params: Optional[str] = 'state=complete',
                 states: Mapping[str, Collection[str]] | None = None,
                 subsets: Mapping[str, Collection[Mapping[str, str]]] | None = None):
        """
        :param data_query_params: the query for the image ids of a training whose loop named no states.
        :param states: `{set: [state]}` of the states each set takes its images from; replaces `data_query_params`.
        :param subsets: `{set: [{uuid, name}]}` of the restricted sets.
        """
        self.data_query_params = data_query_params
        self.data_exchanger = data_exchanger
        self.states = states or {}
        self.subsets = subsets or {}

    async def download_training_data(self, image_folder: str) -> Tuple[List[Dict], int]:
        image_ids = await self._fetch_image_ids()
        image_data, skipped_image_count = await self.download_images_and_annotations(image_ids, image_folder)
        return (image_data, skipped_image_count)

    async def download_images_and_annotations(self, image_ids: List[str], image_folder: str) -> Tuple[List[Dict], int]:
        image_data = await self.data_exchanger.download_images_data(image_ids)
        if self.states or self.subsets:
            image_data = restrict_to_subsets(restrict_to_states(image_data, self.states), self.subsets)
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

    async def _fetch_image_ids(self) -> list[str]:
        if not self.states:
            return await self.data_exchanger.fetch_image_uuids(query_params=self.data_query_params)
        image_ids: list[str] = []
        for state in states_to_fetch(self.states):
            image_ids += await self.data_exchanger.fetch_image_uuids(query_params=f'state={state}')
        return list(dict.fromkeys(image_ids))  # an image that changed its state in between is listed twice
