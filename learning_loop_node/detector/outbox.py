import json
import logging
import os
import shutil
import time
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from glob import glob
from io import BufferedReader, TextIOWrapper
from multiprocessing import Event
from multiprocessing.synchronize import Event as SyncEvent
from threading import Thread
from typing import List, Optional, Tuple, Union

import requests
from fastapi.encoders import jsonable_encoder

from ..data_classes import Detections
from ..globals import GLOBALS
from ..helpers import environment_reader


class OutboxMode(Enum):
    CONTINUOUS_UPLOAD = 'continuous_upload'
    STOPPED = 'stopped'


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
        self.UPLOAD_TIMEOUT_S = 30

        self.shutdown_event: SyncEvent = Event()
        self.upload_process: Optional[Thread] = None

    def save(self, image: bytes, detections: Optional[Detections] = None, tags: Optional[List[str]] = None) -> None:
        if detections is None:
            detections = Detections()
        if not tags:
            tags = []
        identifier = datetime.now().isoformat(sep='_', timespec='milliseconds')
        tmp = f'{GLOBALS.data_folder}/tmp/{identifier}'
        detections.tags = tags
        detections.date = identifier
        os.makedirs(tmp, exist_ok=True)

        with open(tmp + '/image.json', 'w') as f:
            json.dump(jsonable_encoder(asdict(detections)), f)

        with open(tmp + '/image.jpg', 'wb') as f:
            f.write(image)

        if os.path.exists(tmp):
            os.rename(tmp, self.path + '/' + identifier)  # NOTE rename is atomic so upload can run in parallel
        else:
            self.log.error('Could not rename %s to %s', tmp, self.path + '/' + identifier)

    def get_data_files(self):
        return glob(f'{self.path}/*')

    def ensure_continuous_upload(self):
        self.log.debug('start_continuous_upload')
        if self._upload_process_alive():
            self.log.debug('Upload thread already running')
            return

        self.shutdown_event.clear()
        self.upload_process = Thread(target=self._continuous_upload, name='OutboxUpload')
        self.upload_process.start()

    def _continuous_upload(self):
        self.log.info('continuous upload started')
        assert self.shutdown_event is not None
        while not self.shutdown_event.is_set():
            self.upload()
            time.sleep(5)
        self.log.info('continuous upload ended')

    def upload(self):
        items = self.get_data_files()
        if items:
            self.log.info('Found %s images to upload', len(items))
            for i in range(0, len(items), self.BATCH_SIZE):
                batch_items = items[i:i+self.BATCH_SIZE]
                if self.shutdown_event.is_set():
                    break
                try:
                    self._upload_batch(batch_items)
                except Exception:
                    self.log.exception('Could not upload files')
        else:
            self.log.info('No images found to upload')

    def _upload_batch(self, items: List[str]):
        data: List[Tuple[str, Union[TextIOWrapper, BufferedReader]]] = []
        data = [('files', open(f'{item}/image.json', 'r')) for item in items]
        data += [('files', open(f'{item}/image.jpg', 'rb')) for item in items]

        response = requests.post(self.target_uri, files=data, timeout=self.UPLOAD_TIMEOUT_S)
        if response.status_code == 200:
            for item in items:
                shutil.rmtree(item, ignore_errors=True)
            self.log.info('Uploaded %s images successfully', len(items))
        elif response.status_code == 422:
            if len(items) == 1:
                self.log.error('Broken content in image: %s\n Skipping.', items[0])
                shutil.rmtree(items[0], ignore_errors=True)
                return

            self.log.exception('Broken content in batch. Splitting and retrying')
            self._upload_batch(items[:len(items)//2])
            self._upload_batch(items[len(items)//2:])
        else:
            self.log.error('Could not upload images: %s', response.content)

    def ensure_continuous_upload_stopped(self) -> bool:
        self.log.debug('Outbox: Ensuring continuous upload')
        if not self._upload_process_alive():
            self.log.debug('Upload thread already stopped')
            return True
        proc = self.upload_process
        if not proc:
            return True

        try:
            assert self.shutdown_event is not None
            self.shutdown_event.set()
            assert proc is not None
            proc.join(self.UPLOAD_TIMEOUT_S + 1)
        except Exception:
            self.log.exception('Error while shutting down upload thread: ')

        if proc.is_alive():
            self.log.error('Upload thread did not terminate')
            return False

        self.log.info('Upload thread terminated')
        return True

    def _upload_process_alive(self) -> bool:
        return bool(self.upload_process and self.upload_process.is_alive())

    def get_mode(self) -> OutboxMode:
        ''':return: current mode ('continuous_upload' or 'stopped')'''
        if self.upload_process and self.upload_process.is_alive():
            current_mode = OutboxMode.CONTINUOUS_UPLOAD
        else:
            current_mode = OutboxMode.STOPPED

        self.log.debug('Outbox: Current mode is %s', current_mode)
        return current_mode

    def set_mode(self, mode: Union[OutboxMode, str]) -> None:
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
                self.ensure_continuous_upload_stopped()
            except TimeoutError as e:
                raise TimeoutError(f'Upload thread did not terminate within {self.UPLOAD_TIMEOUT_S} seconds.') from e

        self.log.debug('set outbox mode to %s', mode)
