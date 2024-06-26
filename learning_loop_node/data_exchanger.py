import asyncio
import logging
import os
import shutil
import zipfile
from glob import glob
from http import HTTPStatus
from io import BytesIO
from time import time
from typing import Dict, List, Optional

import aiofiles  # type: ignore

from .data_classes import Context
from .helpers.misc import create_resource_paths, create_task, is_valid_image
from .loop_communication import LoopCommunicator
from .trainer.exceptions import CriticalError


class DownloadError(Exception):

    def __init__(self, cause: str, *args: object) -> None:
        super().__init__(*args)
        self.cause = cause

    def __str__(self) -> str:
        return f'DownloadError: {self.cause}'


class DataExchanger():

    def __init__(self, context: Optional[Context], loop_communicator: LoopCommunicator):
        """Exchanges data with the learning loop via the loop_communicator (rest api).

        Args:
            context (Optional[Context]): The context of the node. This is the organization and project name.
            loop_communicator (LoopCommunicator): The loop_communicator to use for communication with the learning loop.

        Note:
            The context can be set later with the set_context method.
        """
        self.set_context(context)
        self.progress = 0.0
        self.loop_communicator = loop_communicator

        self.check_jpeg = shutil.which('jpeginfo') is not None
        if self.check_jpeg:
            logging.info('Detected command line tool "jpeginfo". Images will be checked for validity')
        else:
            logging.error('Missing command line tool "jpeginfo". We cannot check for validity of images.')

    def set_context(self, context: Optional[Context]) -> None:
        self._context = context
        self.progress = 0.0

    @property
    def context(self) -> Context:
        assert self._context, 'DataExchanger: Context was not set yet.. call set_context() first.'
        return self._context

    # ---------------------------- END OF INIT ----------------------------

    async def fetch_image_uuids(self, query_params: Optional[str] = '') -> List[str]:
        """Fetch image uuids from the learning loop data endpoint."""
        logging.info(f'Fetching image uuids for {self.context.organization}/{self.context.project}..')

        response = await self.loop_communicator.get(f'/{self.context.organization}/projects/{self.context.project}/data?{query_params}')
        assert response.status_code == 200, response
        return (response.json())['image_ids']

    async def download_images_data(self, image_uuids: List[str], chunk_size: int = 100) -> List[Dict]:
        """Download image annotations, tags, set and other information for the given image uuids."""
        logging.info(f'Fetching annotations, tags, sets, etc. for {len(image_uuids)} images..')

        num_image_ids = len(image_uuids)
        if num_image_ids == 0:
            logging.info('got empty list. No images were downloaded')
            return []

        progress_factor = 0.5 / num_image_ids  # 50% of progress is for downloading data
        images_data: List[Dict] = []
        for i in range(0, num_image_ids, chunk_size):
            self.progress = i * progress_factor
            chunk_ids = image_uuids[i:i+chunk_size]
            response = await self.loop_communicator.get(f'/{self.context.organization}/projects/{self.context.project}/images?ids={",".join(chunk_ids)}')
            if response.status_code != 200:
                logging.error(f'Error {response.status_code} during downloading image data. Continue with next batch..')
                continue
            images_data += response.json()['images']

        return images_data

    async def download_images(self, image_uuids: List[str], image_folder: str, chunk_size: int = 10) -> None:
        """Downloads images (actual image data). Will skip existing images"""
        logging.info(f'Downloading {len(image_uuids)} images (actual image data).. skipping existing images.')
        if not image_uuids:
            return

        existing_uuids = {os.path.splitext(os.path.basename(image))[0] for image in glob(f'{image_folder}/*.jpg')}
        new_image_uuids = [id for id in image_uuids if id not in existing_uuids]

        paths, _ = create_resource_paths(self.context.organization, self.context.project, new_image_uuids)
        num_image_ids = len(image_uuids)
        os.makedirs(image_folder, exist_ok=True)

        progress_factor = 0.5 / num_image_ids  # second 50% of progress is for downloading images
        for i in range(0, num_image_ids, chunk_size):
            self.progress = 0.5 + i * progress_factor
            chunk_paths = paths[i:i+chunk_size]
            chunk_ids = image_uuids[i:i+chunk_size]
            tasks = []
            for j, chunk_j in enumerate(chunk_paths):
                start = time()
                tasks.append(create_task(self._download_one_image(chunk_j, chunk_ids[j], image_folder)))
                await asyncio.sleep(max(0, 0.02 - (time() - start)))  # prevent too many requests at once
            await asyncio.gather(*tasks)

    async def _download_one_image(self, path: str, image_id: str, image_folder: str) -> None:
        response = await self.loop_communicator.get(path)
        if response.status_code != HTTPStatus.OK:
            logging.error(f'bad status code {response.status_code} for {path}. Details: {response.text}')
            return
        filename = f'{image_folder}/{image_id}.jpg'
        async with aiofiles.open(filename, 'wb') as f:
            await f.write(response.content)
        if not await is_valid_image(filename, self.check_jpeg):
            os.remove(filename)

    async def download_model(self, target_folder: str, context: Context, model_uuid: str, model_format: str) -> List[str]:
        """Downloads a model (and additional meta data like model.json) and returns the paths of the downloaded files.
        Used before training a model (when continuing a finished training) or before detecting images.
        """
        logging.info(f'Downloading model data for uuid {model_uuid} from the loop to {target_folder}..')

        path = f'/{context.organization}/projects/{context.project}/models/{model_uuid}/{model_format}/file'
        response = await self.loop_communicator.get(path, requires_login=False)
        if response.status_code != 200:
            content = response.json()
            logging.error(f'could not download loop/{path}: {response.status_code}, content: {content}')
            raise DownloadError(content['detail'])
        try:
            provided_filename = response.headers.get(
                "Content-Disposition").split("filename=")[1].strip('"')
            content = response.content
        except:
            logging.exception(f'Error during downloading model {path}:')
            raise

        tmp_path = f'/tmp/{os.path.splitext(provided_filename)[0]}'
        shutil.rmtree(tmp_path, ignore_errors=True)
        with zipfile.ZipFile(BytesIO(content), 'r') as zip_:
            zip_.extractall(tmp_path)

        created_files = []
        for file in glob(f'{tmp_path}/**/*', recursive=True):
            new_file = shutil.move(file, target_folder)
            created_files.append(new_file)

        shutil.rmtree(tmp_path, ignore_errors=True)
        logging.info(f'Downloaded model {model_uuid}({model_format}) to {target_folder}.')
        return created_files

    async def upload_model_get_uuid(self, context: Context, files: List[str], training_number: Optional[int], mformat: str) -> str:
        """Used by the trainers. Function returns the new model uuid to use for detection.

        :return: The new model uuid.
        :raise CriticalError: If the upload does not return status code 200.
        """
        response = await self.loop_communicator.put(f'/{context.organization}/projects/{context.project}/trainings/{training_number}/models/latest/{mformat}/file', files=files)
        if response.status_code != 200:
            logging.error(f'Could not upload model for training {training_number}, format {mformat}: {response.text}')
            raise CriticalError(
                f'Could not upload model for training {training_number}, format {mformat}: {response.text}')

        uploaded_model = response.json()
        logging.info(f'Uploaded model for training {training_number}, format {mformat}. Response is: {uploaded_model}')
        return uploaded_model['id']
