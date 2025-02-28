import asyncio
import io
import json
import logging
import os
import shutil
from asyncio import Task
from dataclasses import asdict
from datetime import datetime
from glob import glob
from io import BufferedReader, TextIOWrapper
from multiprocessing import Event
from multiprocessing.synchronize import Event as SyncEvent
from typing import List, Optional, Tuple, Union

import aiohttp
import PIL
import PIL.Image  # type: ignore
from fastapi.encoders import jsonable_encoder

from ..data_classes import ImageMetadata
from ..enums import OutboxMode
from ..globals import GLOBALS
from ..helpers import environment_reader


class Outbox():
    def __init__(self) -> None:
        self.log = logging.getLogger()
        self.path = f'{GLOBALS.data_folder}/outbox'
        os.makedirs(self.path, exist_ok=True)

        self.log = logging.getLogger()
        host = environment_reader.host()
        o = environment_reader.organization()
        p = environment_reader.project()

        assert o and p, 'Outbox needs an organization and a project '
        base_url = f'http{"s" if "learning-loop.ai" in host else ""}://{host}/api'
        base: str = base_url
        self.target_uri = f'{base}/{o}/projects/{p}/images'
        self.log.info('Outbox initialized with target_uri: %s', self.target_uri)

        self.BATCH_SIZE = 20
        self.MAX_UPLOAD_LENGTH = 1000  # only affects the `upload_folders` list
        self.UPLOAD_INTERVAL_S = 5
        self.UPLOAD_TIMEOUT_S = 30

        self.shutdown_event: SyncEvent = Event()
        self.upload_task: Optional[Task] = None

        self.upload_counter = 0

        self.priority_upload_folders: List[str] = []
        self.upload_folders: List[str] = []

        self.upload_folders = self.get_all_data_files()  # make sure to upload all existing images (e.g. after a restart)

    def save(self,
             image: bytes,
             image_metadata: Optional[ImageMetadata] = None,
             tags: Optional[List[str]] = None,
             source: Optional[str] = None,
             creation_date: Optional[str] = None,
             upload_priority: bool = False) -> None:

        if not self._is_valid_jpg(image):
            self.log.error('Invalid jpg image')
            return

        if image_metadata is None:
            image_metadata = ImageMetadata()
        if not tags:
            tags = []
        identifier = datetime.now().isoformat(sep='_', timespec='microseconds')
        if os.path.exists(self.path + '/' + identifier):
            self.log.error('Directory with identifier %s already exists', identifier)
            return
        tmp = f'{GLOBALS.data_folder}/tmp/{identifier}'
        image_metadata.tags = tags
        if self._is_valid_isoformat(creation_date):
            image_metadata.created = creation_date
        else:
            image_metadata.created = identifier

        image_metadata.source = source or 'unknown'
        os.makedirs(tmp, exist_ok=True)

        with open(tmp + f'/image_{identifier}.json', 'w') as f:
            json.dump(jsonable_encoder(asdict(image_metadata)), f)

        with open(tmp + f'/image_{identifier}.jpg', 'wb') as f:
            f.write(image)

        if os.path.exists(tmp):
            os.rename(tmp, self.path + '/' + identifier)  # NOTE rename is atomic so upload can run in parallel
        else:
            self.log.error('Could not rename %s to %s', tmp, self.path + '/' + identifier)

        if upload_priority:
            self.priority_upload_folders.append(self.path + '/' + identifier)
        else:
            self.upload_folders.insert(0, self.path + '/' + identifier)

        # Cut off the upload list if it gets too long
        if len(self.upload_folders) > self.MAX_UPLOAD_LENGTH:
            items_to_drop = self.upload_folders[self.MAX_UPLOAD_LENGTH:]
            self.log.info('Dropping %s images from upload list', len(items_to_drop))
            try:
                for item in items_to_drop:
                    shutil.rmtree(item)
                    self.log.debug('Deleted %s', item)
                self.upload_folders = self.upload_folders[:self.MAX_UPLOAD_LENGTH]
            except Exception:
                self.log.exception('Failed to cut upload list')

    def _is_valid_isoformat(self, date: Optional[str]) -> bool:
        if date is None:
            return False
        try:
            datetime.fromisoformat(date)
            return True
        except Exception:
            return False

    def get_all_data_files(self) -> List[str]:
        return glob(f'{self.path}/*')

    def get_upload_folders(self) -> List[str]:
        return self.priority_upload_folders + self.upload_folders

    def ensure_continuous_upload(self) -> None:
        self.log.debug('start_continuous_upload')
        if self._upload_process_alive():
            self.log.debug('Upload thread already running')
            return

        self.shutdown_event.clear()
        self.upload_task = asyncio.create_task(self._continuous_upload())

    async def _continuous_upload(self) -> None:
        self.log.info('continuous upload started')
        assert self.shutdown_event is not None
        while not self.shutdown_event.is_set():
            await self.upload()
            await asyncio.sleep(self.UPLOAD_INTERVAL_S)
        self.log.info('continuous upload ended')

    async def upload(self) -> None:
        items = self.get_upload_folders()
        if not items:
            self.log.debug('No images found to upload')
            return

        self.log.info('Found %s images to upload', len(items))
        # NOTE (for reviewer):
        # I changed the behaviour from trying to clear the outbox in each upload cycle to uploading the first BS images in each 5-sec cycle
        # This simplifies the code and has the advantage that newer images or manual uploads are uploaded earlier

        batch_items = items[:self.BATCH_SIZE]
        try:
            await self._upload_batch(batch_items)
        except Exception:
            self.log.exception('Could not upload files')

    async def _upload_batch(self, items: List[str]) -> None:
        """
        Uploads a batch of images to the server.
        :param items: List of folders to upload (each folder contains an image and a metadata file)
        """

        # NOTE: keys are not relevant for the server, but using a fixed key like 'files'
        # results in a post failure on the first run of the test in a docker environment (WTF)

        data: List[Tuple[str, Union[TextIOWrapper, BufferedReader]]] = []
        for item in items:
            identifier = os.path.basename(item)
            data.append(('files', open(f'{item}/image_{identifier}.json', 'r')))
            data.append(('files', open(f'{item}/image_{identifier}.jpg', 'rb')))

        try:
            async with aiohttp.ClientSession() as session:
                response = await session.post(self.target_uri, data=data, timeout=aiohttp.ClientTimeout(total=self.UPLOAD_TIMEOUT_S))
        except Exception:
            self.log.exception('Could not upload images')
            return
        finally:
            self.log.debug('Closing files')
            for _, file in data:
                file.close()

        if response.status == 200:
            self.upload_counter += len(items)
            for item in items:
                try:
                    shutil.rmtree(item)
                    self.log.debug('Deleted %s', item)
                except Exception:
                    self.log.exception('Failed to delete %s', item)
            self.log.info('Uploaded %s images successfully', len(items))

        elif response.status == 422:
            if len(items) == 1:
                self.log.error('Broken content in image: %s\n Skipping.', items[0])
                shutil.rmtree(items[0], ignore_errors=True)
                return

            self.log.exception('Broken content in batch. Splitting and retrying')
            await self._upload_batch(items[:len(items)//2])
            await self._upload_batch(items[len(items)//2:])
        elif response.status == 429:
            self.log.warning('Too many requests: %s', response.content)
        else:
            self.log.error('Could not upload images: %s', response.content)

    def _is_valid_jpg(self, image: bytes) -> bool:
        try:
            _ = PIL.Image.open(io.BytesIO(image), formats=['JPEG'])
            return True
        except Exception:
            self.log.exception('Invalid jpg image')
            return False

    async def ensure_continuous_upload_stopped(self) -> bool:
        self.log.debug('Outbox: Ensuring continuous upload')
        if not self._upload_process_alive():
            self.log.debug('Upload thread already stopped')
            return True

        if not self.upload_task:
            return True

        try:
            assert self.shutdown_event is not None
            self.shutdown_event.set()
            await asyncio.wait_for(self.upload_task, timeout=self.UPLOAD_TIMEOUT_S + 1)
        except asyncio.TimeoutError:
            self.log.error('Upload task did not terminate in time')
            return False
        except Exception:
            self.log.exception('Error while shutting down upload task: ')
            return False

        self.log.info('Upload thread terminated')
        return True

    def _upload_process_alive(self) -> bool:
        return bool(self.upload_task and not self.upload_task.done())

    def get_mode(self) -> OutboxMode:
        ''':return: current mode ('continuous_upload' or 'stopped')'''
        if self._upload_process_alive():
            current_mode = OutboxMode.CONTINUOUS_UPLOAD
        else:
            current_mode = OutboxMode.STOPPED

        self.log.debug('Outbox: Current mode is %s', current_mode)
        return current_mode

    async def set_mode(self, mode: Union[OutboxMode, str]) -> None:
        ''':param mode: 'continuous_upload' or 'stopped'
        :raises ValueError: if mode is not a valid OutboxMode
        :raises TimeoutError: if the upload thread does not terminate within 31 seconds with mode='stopped'
        '''
        if isinstance(mode, str):
            mode = OutboxMode(mode)

        if mode == OutboxMode.CONTINUOUS_UPLOAD:
            self.ensure_continuous_upload()
        elif mode == OutboxMode.STOPPED:
            try:
                await self.ensure_continuous_upload_stopped()
            except TimeoutError as e:
                raise TimeoutError(f'Upload thread did not terminate within {self.UPLOAD_TIMEOUT_S} seconds.') from e

        self.log.debug('set outbox mode to %s', mode)
