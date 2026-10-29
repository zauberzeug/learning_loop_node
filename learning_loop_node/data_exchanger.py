import asyncio
import logging
import os
import shutil
import zipfile
from glob import glob
from http import HTTPStatus
from io import BytesIO
from time import time
from typing import Any, Dict, List, Optional

import aiofiles  # type: ignore

from .data_classes import Context
from .helpers.misc import create_resource_paths, create_task, is_valid_image, is_valid_uuid4
from .loop_communication import LoopCommunicator
from .trainer.exceptions import CriticalError, ModelUploadFailedError, TrainingResultsUploadFailedError

logger = logging.getLogger(__name__)


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
        logging.info('Fetching image uuids for %s/%s..', self.context.organization, self.context.project)

        response = await self.loop_communicator.get(f'/{self.context.organization}/projects/{self.context.project}/data?{query_params}')
        assert response.status_code == 200, response
        return (response.json())['image_ids']

    async def download_images_data(self, image_uuids: List[str], chunk_size: int = 100) -> List[Dict[str, Any]]:
        """Download image annotations, tags, set and other information for the given image uuids."""
        logging.info('Fetching annotations, tags, sets, etc. for %s images..', len(image_uuids))

        num_image_ids = len(image_uuids)
        if num_image_ids == 0:
            logging.info('got empty list. No images were downloaded')
            return []

        progress_factor = 0.5 / num_image_ids  # first 50% of progress is for downloading data
        images_data: List[Dict[str, Any]] = []
        for i in range(0, num_image_ids, chunk_size):
            self.progress = i * progress_factor
            chunk_ids = image_uuids[i:i+chunk_size]
            response = await self.loop_communicator.get(f'/{self.context.organization}/projects/{self.context.project}/images?ids={",".join(chunk_ids)}')
            if response.status_code != 200:
                logging.error('Error %s during downloading image data. Continue with next batch..', response.status_code)
                continue
            images_data += response.json()['images']

        return images_data

    async def download_images(self, image_uuids: List[str], image_folder: str, chunk_size: int = 10) -> None:
        """Downloads images (actual image data). Will skip existing images"""
        logging.info('Downloading %s images (actual image data).. skipping existing images.', len(image_uuids))
        if not image_uuids:
            return

        existing_uuids = {os.path.splitext(os.path.basename(image))[0] for image in glob(f'{image_folder}/*.jpg')}
        new_image_uuids = [id for id in image_uuids if id not in existing_uuids]

        paths, _ = create_resource_paths(self.context.organization, self.context.project, new_image_uuids)
        num_new_image_ids = len(new_image_uuids)
        if num_new_image_ids == 0:
            logging.info('All images are already downloaded. Nothing to do.')
            self.progress = 1.0
            return

        logging.info('Downloading %s new images to %s..', num_new_image_ids, image_folder)
        os.makedirs(image_folder, exist_ok=True)

        progress_factor = 0.5 / num_new_image_ids  # second 50% of progress is for downloading images
        for i in range(0, num_new_image_ids, chunk_size):
            self.progress = 0.5 + i * progress_factor
            chunk_paths = paths[i:i+chunk_size]
            chunk_ids = new_image_uuids[i:i+chunk_size]
            tasks = []
            for j, chunk_j in enumerate(chunk_paths):
                start = time()
                tasks.append(create_task(self._download_one_image(chunk_j, chunk_ids[j], image_folder)))
                await asyncio.sleep(max(0, 0.02 - (time() - start)))  # prevent too many requests at once
            await asyncio.gather(*tasks)
        self.progress = 1.0

    async def _download_one_image(self, path: str, image_id: str, image_folder: str) -> None:
        response = await self.loop_communicator.get(path)
        if response.status_code != HTTPStatus.OK:
            logging.error('bad status code %s for %s. Details: %s', response.status_code, path, response.text)
            return
        filename = f'{image_folder}/{image_id}.jpg'
        async with aiofiles.open(filename, 'wb') as f:
            await f.write(response.content)
        if not await is_valid_image(filename, self.check_jpeg):
            logging.error('Invalid image "%s". Removing it..', filename)
            os.remove(filename)
        else:
            logging.debug('Downloaded image "%s"', filename)

    async def download_model(self, target_folder: str, context: Context, model_uuid: str, model_format: str) -> List[str]:
        """Downloads a model (and additional meta data like model.json) and returns the paths of the downloaded files.
        Used before training a model (when continuing a finished training) or before detecting images.
        """
        logging.info('Downloading model data for uuid %s from the loop to %s..', model_uuid, target_folder)

        path = f'/{context.organization}/projects/{context.project}/models/{model_uuid}/{model_format}/file'
        response = await self.loop_communicator.get(path, requires_login=False, timeout=60*10)
        if response.status_code != 200:
            decoded_content = response.content.decode('utf-8')
            logging.error('could not download loop/%s: %s, content: %s', path,
                          response.status_code, decoded_content)
            raise DownloadError(decoded_content)
        try:
            provided_filename = response.headers.get(
                "Content-Disposition").split("filename=")[1].strip('"')
            content = response.content
        except:
            logging.exception('Error while downloading model %s:', path)
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
        logging.info('Downloaded model %s(%s) to %s.', model_uuid, model_format, target_folder)
        return created_files

    async def upload_model_get_uuid(self, context: Context, files: List[str], training_number: Optional[int], mformat: str) -> str | None:
        """Used by the trainers. Function returns the new model uuid to use for detection.
        A loop that already stores the model in this format answers 409 and should name the model's uuid there.

        :return: The new model uuid, or None if the loop already stores the model in this format but names no valid uuid.
        :raise CriticalError: If the loop rejects the upload with a client error (other than 408 and 429).
        :raise ModelUploadFailedError: If the loop could not store the model, e.g. on a server error.
        """
        path = f'/{context.organization}/projects/{context.project}/trainings/{training_number}/models/latest/{mformat}/file'
        response = await self.loop_communicator.put(path, files=files)
        if response.status_code == HTTPStatus.CONFLICT:
            try:
                stored_model_id = response.json()['detail']['id']
            except (ValueError, KeyError, TypeError):  # older loops name no id
                stored_model_id = None
            if not is_valid_uuid4(stored_model_id):
                logger.warning('The loop already stores the model of training %s in format %s, but names no id',
                               training_number, mformat)
                return None
            logger.info('The loop already stores model %s of training %s in format %s',
                        stored_model_id, training_number, mformat)
            return stored_model_id
        if response.status_code != HTTPStatus.OK:
            logger.error('Could not upload model for training %s, format %s: %s %s',
                         training_number, mformat, response.status_code, response.text)
            error = CriticalError if _rejects_upload(response.status_code) else ModelUploadFailedError
            raise error(
                f'Could not upload model for training {training_number}, format {mformat}: {response.status_code} {response.text}')

        uploaded_model = response.json()
        logger.info('Uploaded model for training %s, format %s. Response is: %s',
                    training_number, mformat, uploaded_model)
        return uploaded_model['id']

    async def upload_training_results(self, context: Context, training_number: int, part: str, zip_path: str) -> None:
        """Store one part of a training's results beside its latest model; a part sent again replaces the stored one.

        The results are optional, so an answer from the loop is final, whatever it says; only an upload that got
        no answer is worth sending again.

        :raise TrainingResultsUploadFailedError: If the loop answered with anything but success (other than 408).
        :raise ModelUploadFailedError: If the loop answered 408, having given up waiting for the upload.
        """
        path = (f'/{context.organization}/projects/{context.project}/trainings/{training_number}'
                f'/training_results/{part}')
        response = await self.loop_communicator.put(path, files=[zip_path], timeout=60*10)
        if response.status_code == HTTPStatus.REQUEST_TIMEOUT:
            raise ModelUploadFailedError(f'The loop gave up waiting for the {part} results of training '
                                         f'{training_number}: {response.status_code} {response.text}')
        if response.status_code != HTTPStatus.OK:
            raise TrainingResultsUploadFailedError(f'The loop did not store the {part} results of training '
                                                   f'{training_number}: {response.status_code} {response.text}')
        logger.info('Uploaded the %s results of training %s', part, training_number)


def _rejects_upload(status_code: int) -> bool:
    """A client error other than 408 and 429 answers the same however often the upload is sent."""
    return 400 <= status_code < 500 and status_code not in (HTTPStatus.REQUEST_TIMEOUT, HTTPStatus.TOO_MANY_REQUESTS)
