import asyncio
import logging
import os
import shutil
import time
import zipfile
from glob import glob
from http import HTTPStatus
from io import BytesIO
from time import perf_counter
from typing import Dict, List, Optional

import aiofiles
from tqdm.asyncio import tqdm

from .data_classes import Context
from .helpers.misc import create_resource_paths, create_task
from .loop_communication import LoopCommunicator

check_jpeg = shutil.which('jpeginfo') is not None


class DownloadError(Exception):

    def __init__(self, cause: str, *args: object) -> None:
        super().__init__(*args)
        self.cause = cause


class DataExchanger():

    def __init__(self, context: Optional[Context], loop_communicator: LoopCommunicator):
        self.context = context
        self.loop_communicator = loop_communicator
        self.progress = 0.0

    def set_context(self, context: Context):
        self.context = context

    async def fetch_image_ids(self, query_params: Optional[str] = '') -> List[str]:
        if self.context is None:
            logging.warning('context was not set yet')
            return []

        response = await self.loop_communicator.get(f'/{self.context.organization}/projects/{self.context.project}/data?{query_params}')
        assert response.status_code == 200, response
        return (response.json())['image_ids']

    async def download_images_data(self, ids: List[str]) -> List[Dict]:
        '''Download image annotations etc.'''
        if self.context is None:
            logging.warning('context was not set yet')
            return []

        return await self._download_images_data(self.context.organization, self.context.project, ids)

    async def download_images(self, image_ids: List[str], image_folder: str) -> None:
        '''Download images. Will skip existing images'''
        if self.context is None:
            logging.warning('context was not set yet')
            return

        await self.delete_corrupt_images(image_folder)
        new_image_ids = await asyncio.get_event_loop().run_in_executor(None, DataExchanger.filter_existing_images, image_ids, image_folder)
        paths, ids = create_resource_paths(self.context.organization, self.context.project, new_image_ids)
        await self._download_images(paths, ids, image_folder)

    @staticmethod
    async def delete_corrupt_images(image_folder: str) -> None:
        logging.info('deleting corrupt images')
        n_deleted = 0
        for image in glob(f'{image_folder}/*.jpg'):
            if not await DataExchanger.is_valid_image(image):
                logging.debug(f'  deleting image {image}')
                os.remove(image)
                n_deleted += 1

        logging.info(f'deleted {n_deleted} images')

    @staticmethod
    def filter_existing_images(all_image_ids, image_folder) -> List[str]:
        logging.info(f'### Going to filter {len(all_image_ids)} images ids')
        start = perf_counter()
        ids = [os.path.splitext(os.path.basename(image))[0]
               for image in glob(f'{image_folder}/*.jpg')]
        logging.info(f'found {len(ids)} images on disc')
        result = [id for id in all_image_ids if id not in ids]
        end = perf_counter()
        logging.info(f'calculated {len(result)} new image ids, which took {end-start:0.2f} seconds')
        return result

    def jepeg_check_info(self):
        if check_jpeg:
            logging.info('Detected command line tool "jpeginfo". Images will be checked for validity')
        else:
            logging.error('Missing command line tool "jpeginfo". We can not check for validity of images.')

    async def _download_images_data(self, organization: str, project: str, image_ids: List[str], chunk_size: int = 100) -> List[Dict]:
        logging.info('fetching annotations and other image data')
        num_image_ids = len(image_ids)
        self.jepeg_check_info()
        images_data = []
        if num_image_ids == 0:
            logging.info('got empty list. No images were downloaded')
            return images_data
        starttime = time.time()
        progress_factor = 0.5 / num_image_ids  # 50% of progress is for downloading data
        for i in tqdm(range(0, num_image_ids, chunk_size), position=0, leave=True):
            self.progress = i * progress_factor
            chunk_ids = image_ids[i:i+chunk_size]
            response = await self.loop_communicator.get(f'/{organization}/projects/{project}/images?ids={",".join(chunk_ids)}')
            if response.status_code != 200:
                logging.error(
                    f'Error during downloading list of images. Statuscode is {response.status_code}')
                continue
            images_data += response.json()['images']
            total_time = round(time.time() - starttime, 1)
            if images_data:
                per100 = total_time / len(images_data) * 100
                logging.debug(f'[+] Performance: {total_time} sec total. Per 100 : {per100:.1f} sec')
            else:
                logging.debug(f'[+] Performance: {total_time} sec total.')
        return images_data

    async def _download_images(self, paths: List[str], image_ids: List[str], image_folder: str, chunk_size: int = 10) -> None:
        num_image_ids = len(image_ids)
        if num_image_ids == 0:
            logging.debug('got empty list. No images were downloaded')
            return
        logging.info('fetching image files')
        starttime = time.time()
        os.makedirs(image_folder, exist_ok=True)

        progress_factor = 0.5 / num_image_ids  # second 50% of progress is for downloading images
        for i in tqdm(range(0, num_image_ids, chunk_size), position=0, leave=True):
            self.progress = 0.5 + i * progress_factor
            chunk_paths = paths[i:i+chunk_size]
            chunk_ids = image_ids[i:i+chunk_size]
            tasks = []
            for j, chunk_j in enumerate(chunk_paths):
                tasks.append(create_task(self.download_one_image(chunk_j, chunk_ids[j], image_folder)))
            await asyncio.gather(*tasks)
            total_time = round(time.time() - starttime, 1)
            per100 = total_time / (i + len(tasks)) * 100
            logging.debug(f'[+] Performance (image files): {total_time} sec total. Per 100 : {per100:.1f}')

    async def download_one_image(self, path: str, image_id: str, image_folder: str) -> None:
        response = await self.loop_communicator.get(path)
        if response.status_code != HTTPStatus.OK:
            logging.error(f'bad status code {response.status_code} for {path}')
            return
        filename = f'{image_folder}/{image_id}.jpg'
        async with aiofiles.open(filename, 'wb') as f:
            await f.write(response.content)
        if not await self.is_valid_image(filename):
            os.remove(filename)

    @staticmethod
    async def is_valid_image(filename: str) -> bool:
        if not os.path.isfile(filename) or os.path.getsize(filename) == 0:
            return False
        if not check_jpeg:
            return True

        info = await asyncio.create_subprocess_shell(
            f'jpeginfo -c {filename}',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        out, _ = await info.communicate()
        return "OK" in out.decode()

    async def download_model(self, target_folder: str, context: Context, model_id: str, model_format: str) -> List[str]:
        path = f'/{context.organization}/projects/{context.project}/models/{model_id}/{model_format}/file'
        response = await self.loop_communicator.get(path, requires_login=False)
        if response.status_code != 200:
            content = response.json()
            logging.error(
                f'could not download {self.loop_communicator.base_url}/{path}: {response.status_code}, content: {content}')
            raise DownloadError(content['detail'])
        try:
            provided_filename = response.headers.get(
                "Content-Disposition").split("filename=")[1].strip('"')
            content = response.content
        except:
            logging.error(f'Error during downloading model {path}:')
            try:
                logging.exception(response.json())
            except Exception:
                pass
            raise

        # unzip and place downloaded model
        tmp_path = f'/tmp/{os.path.splitext(provided_filename)[0]}'
        shutil.rmtree(tmp_path, ignore_errors=True)
        with zipfile.ZipFile(BytesIO(content), 'r') as zip_:
            zip_.extractall(tmp_path)

        logging.info(f'---- downloaded model {model_id} to {tmp_path}.')

        created_files = []
        files = glob(f'{tmp_path}/**/*', recursive=True)
        for file in files:
            new_file = shutil.move(file, target_folder)
            logging.info(f'moved model file {os.path.basename(file)} to {new_file}.')
            created_files.append(new_file)
        return created_files

    async def upload_model(self, context: Context, files: List[str], model_id: str, mformat: str) -> None:
        response = await self.loop_communicator.put(f'/{context.organization}/projects/{context.project}/models/{model_id}/{mformat}/file', files=files)
        if response.status_code != 200:
            msg = f'---- could not upload model with id {model_id} and format {mformat}. Details: {response.text}'
            raise Exception(msg)
        logging.info(f'---- uploaded model with id {model_id} and format {mformat}.')

    async def upload_model_for_training(self, context: Context, files: List[str], training_number: Optional[int], mformat: str) -> Optional[str]:
        """Returns the new model uuid to use for detection."""
        response = await self.loop_communicator.put(f'/{context.organization}/projects/{context.project}/trainings/{training_number}/models/latest/{mformat}/file', files=files)
        if response.status_code != 200:
            msg = f'---- could not upload model for training {training_number} and format {mformat}. Details: {response.text}'
            logging.error(msg)
            response.raise_for_status()
            return None
        else:
            uploaded_model = response.json()
            logging.info(
                f'---- uploaded model for training {training_number} and format {mformat}. Model id is {uploaded_model}')
            return uploaded_model['id']
