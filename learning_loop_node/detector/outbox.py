import asyncio
import io
import json
import logging
import os
import shutil
from asyncio import Task
from collections import deque
from dataclasses import asdict
from datetime import datetime
from glob import glob
from io import BufferedReader, TextIOWrapper
from multiprocessing import Event
from multiprocessing.synchronize import Event as SyncEvent
from threading import Lock
from typing import List, Optional, Tuple, TypeVar, Union

import aiohttp
import numpy as np
import PIL
import PIL.Image  # type: ignore
from fastapi.encoders import jsonable_encoder

from ..data_classes import ImageMetadata
from ..enums import OutboxMode
from ..globals import GLOBALS
from ..helpers import environment_reader, run
from ..helpers.misc import numpy_array_to_jpg_bytes

T = TypeVar('T')


class Outbox():
    """
    Outbox is a class that handles the uploading of images to the learning loop.
    It uploads images from an internal queue (lifo) in batches of 20 every 5 seconds.
    It handles upload failures by splitting the upload into two smaller batches until the problematic image is identified - and removed.
    Any image can be saved to the normal or the priority queue.
    Images in the priority queue are uploaded first.
    The total queue length is limited to 1000 images.
    """

    def __init__(self) -> None:
        self.log = logging.getLogger()
        self.path = f'{GLOBALS.data_folder}/outbox'
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(f'{self.path}/priority', exist_ok=True)
        os.makedirs(f'{self.path}/normal', exist_ok=True)

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
        self.upload_folders: deque[str] = deque()
        self.folders_lock = Lock()

        for file in glob(f'{self.path}/priority/*'):
            self.priority_upload_folders.append(file)
        for file in glob(f'{self.path}/normal/*'):
            self.upload_folders.append(file)

    async def save(self,
                   image: np.ndarray,
                   image_metadata: Optional[ImageMetadata] = None,
                   upload_priority: bool = False) -> None:
        """
        Save an image and its metadata to disk. 

        The data will be picked up by the continuous upload process.
        """

        jpg_bytes = numpy_array_to_jpg_bytes(image)

        if image_metadata is None:
            image_metadata = ImageMetadata()

        identifier = datetime.now().isoformat(sep='_', timespec='microseconds')

        try:
            await run.io_bound(self._save_files_to_disk, identifier, jpg_bytes, image_metadata, upload_priority)
        except Exception as e:
            self.log.error('Failed to save files for image %s: %s', identifier, e)
            return

        if upload_priority:
            self.priority_upload_folders.append(f'{self.path}/priority/{identifier}')
        else:
            self.upload_folders.appendleft(f'{self.path}/normal/{identifier}')

        await self._trim_upload_queue()

    def _save_files_to_disk(self,
                            identifier: str,
                            jpeg_image: bytes,
                            image_metadata: ImageMetadata,
                            upload_priority: bool) -> None:
        subpath = 'priority' if upload_priority else 'normal'
        full_path = f'{self.path}/{subpath}/{identifier}'
        if os.path.exists(full_path):
            raise FileExistsError(f'Directory with identifier {identifier} already exists')

        tmp = f'{GLOBALS.data_folder}/tmp/{identifier}'
        os.makedirs(tmp, exist_ok=True)

        with open(tmp + f'/image_{identifier}.json', 'w') as f:
            json.dump(jsonable_encoder(asdict(image_metadata)), f)

        with open(tmp + f'/image_{identifier}.jpg', 'wb') as f:
            f.write(jpeg_image)

        if not os.path.exists(tmp):
            self.log.error('Could not rename %s to %s', tmp, full_path)
            raise FileNotFoundError(f'Could not rename {tmp} to {full_path}')

        os.rename(tmp, full_path)

    async def _trim_upload_queue(self) -> None:
        if len(self.upload_folders) > self.MAX_UPLOAD_LENGTH:
            excess = len(self.upload_folders) - self.MAX_UPLOAD_LENGTH
            self.log.info('Dropping %s images from upload list', excess)

            folders_to_delete = []
            for _ in range(excess):
                if self.upload_folders:
                    try:
                        folder = self.upload_folders.pop()
                        folders_to_delete.append(folder)
                    except Exception:
                        self.log.exception('Failed to get item from upload_folders')

            await run.io_bound(self._delete_folders, folders_to_delete)

    def _delete_folders(self, folders_to_delete: List[str]) -> None:
        for folder in folders_to_delete:
            try:
                shutil.rmtree(folder)
                self.log.debug('Deleted %s', folder)
            except Exception:
                self.log.exception('Failed to delete %s', folder)

    def _is_valid_isoformat(self, date: Optional[str]) -> bool:
        if date is None:
            return False
        try:
            datetime.fromisoformat(date)
            return True
        except Exception:
            return False

    def get_upload_folders(self) -> List[str]:
        with self.folders_lock:
            return self.priority_upload_folders + list(self.upload_folders)

    def ensure_continuous_upload(self) -> None:
        self.log.debug('start_continuous_upload')
        if self._upload_process_alive():
            self.log.debug('Upload thread already running')
            return

        self.shutdown_event.clear()
        self.upload_task = asyncio.create_task(self._continuous_upload())

    async def _continuous_upload(self) -> None:
        self.log.info('continuous upload started')
        assert self.shutdown_event is not None, 'shutdown_event is None'
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

        batch_items = items[:self.BATCH_SIZE]
        try:
            await self._upload_batch(batch_items)
        except Exception:
            self.log.exception('Could not upload files')

    async def _clear_item(self, item: str) -> None:
        try:
            if item in self.upload_folders:
                self.upload_folders.remove(item)
            if item in self.priority_upload_folders:
                self.priority_upload_folders.remove(item)
            await run.io_bound(shutil.rmtree, item, ignore_errors=True)
            self.log.debug('Deleted %s', item)
        except Exception:
            self.log.exception('Failed to delete %s', item)

    async def _upload_batch(self, items: List[str]) -> None:
        """
        Uploads a batch of images to the server.
        :param items: List of folders to upload (each folder contains an image and a metadata file)
        """

        data: List[Tuple[str, Union[TextIOWrapper, BufferedReader]]] = []
        for item in items:
            if not os.path.exists(item):
                await self._clear_item(item)
                continue
            identifier = os.path.basename(item)
            data.append(('files', open(f'{item}/image_{identifier}.json', 'r')))
            data.append(('files', open(f'{item}/image_{identifier}.jpg', 'rb')))

        try:
            async with aiohttp.ClientSession() as session:
                response = await session.post(self.target_uri, data=data, timeout=aiohttp.ClientTimeout(total=self.UPLOAD_TIMEOUT_S))
                await response.read()
        except Exception:
            self.log.exception('Could not upload images')
            return
        finally:
            self.log.debug('Closing files')
            for _, file in data:
                file.close()

        if response.status == 200:
            self.upload_counter += len(items)
            self.log.debug('Uploaded %s images', len(items))
            for item in items:
                await self._clear_item(item)
            self.log.debug('Cleared %s images', len(items))
            return

        if response.status == 422:
            if len(items) == 1:
                self.log.error('Broken content in image: %s\n Skipping.', items[0])
                await self._clear_item(items[0])
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
            assert self.shutdown_event is not None, 'shutdown_event is None'
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
