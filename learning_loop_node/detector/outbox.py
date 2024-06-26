import json
import logging
import os
import shutil
import time
from dataclasses import asdict
from datetime import datetime
from enum import StrEnum
from glob import glob
from multiprocessing import Event
from multiprocessing.synchronize import Event as SyncEvent
from threading import Thread
from typing import List, Optional

import requests
from fastapi.encoders import jsonable_encoder

from ..data_classes import Detections
from ..globals import GLOBALS
from ..helpers import environment_reader


class OutboxMode(StrEnum):
    CONTINUOUS_UPLOAD = 'continuous_upload'
    STOPPED = 'stopped'


class Outbox():

    def __init__(self) -> None:
        self.log = logging.getLogger()
        self.path = f'{GLOBALS.data_folder}/outbox'
        os.makedirs(self.path, exist_ok=True)

        host = environment_reader.host()
        o = environment_reader.organization()
        p = environment_reader.project()

        assert o and p, 'Outbox needs an organization and a project '
        base_url = f'http{"s" if "learning-loop.ai" in host else ""}://{host}/api'
        base: str = base_url
        self.target_uri = f'{base}/{o}/projects/{p}/images'
        self.log.info('Outbox initialized with target_uri: %s', self.target_uri)

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

    def start_continuous_upload(self):
        self.upload_process = Thread(target=self._continuous_upload)
        self.upload_process.start()

    def _continuous_upload(self):
        self.log.info('start continuous upload')
        assert self.shutdown_event is not None
        while not self.shutdown_event.is_set():
            self.upload()
            time.sleep(1)
        self.log.info('stop continuous upload')

    def upload(self):
        items = self.get_data_files()
        if items:
            self.log.info('Found %s images to upload', len(items))
        for item in items:
            if self.shutdown_event.is_set():
                break
            try:
                data = [('files', open(f'{item}/image.json', 'r')),
                        ('files', open(f'{item}/image.jpg', 'rb'))]

                response = requests.post(self.target_uri, files=data, timeout=30)
                if response.status_code == 200:
                    shutil.rmtree(item)
                    self.log.info('uploaded %s successfully', item)
                elif response.status_code == 422:
                    self.log.error('Broken content in %s: dropping this data', item)
                    shutil.rmtree(item)
                else:
                    self.log.error('Could not upload %s: %s', item, response.status_code)
            except Exception:
                self.log.exception('could not upload files')

    def stop_continuous_upload(self, timeout=31):
        proc = self.upload_process
        if not proc:
            return

        try:
            assert self.shutdown_event is not None
            self.shutdown_event.set()
            assert proc is not None
            proc.join(timeout)
        except Exception:
            logging.exception('Error while shutting down upload thread')

        if proc.is_alive():
            self.log.error('Upload thread did not terminate')

    def get_mode(self) -> OutboxMode:
        ''':return: current mode ('continuous_upload' or 'stopped')'''
        if self.upload_process and self.upload_process.is_alive():
            return OutboxMode.CONTINUOUS_UPLOAD
        return OutboxMode.STOPPED

    def set_mode(self, mode: OutboxMode | str) -> None:
        ''':param mode: 'continuous_upload' or 'stopped'
        :raises ValueError: if mode is not a valid OutboxMode
        '''
        if isinstance(mode, str):
            mode = OutboxMode(mode)

        if mode == OutboxMode.CONTINUOUS_UPLOAD:
            self.start_continuous_upload()
        elif mode == OutboxMode.STOPPED:
            self.stop_continuous_upload()
