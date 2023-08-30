import json
import logging
import os
import shutil
import time
from datetime import datetime
from glob import glob
from multiprocessing import Event
from threading import Thread
from typing import List

import requests
from fastapi.encoders import jsonable_encoder

from learning_loop_node import environment_reader
from learning_loop_node.detector.detections import Detections
from learning_loop_node.globals import GLOBALS


class Outbox():

    def __init__(self) -> None:
        self.log = logging.getLogger()
        self.path = f'{GLOBALS.data_folder}/outbox'
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        host = os.environ.get('HOST', 'learning-loop.ai')
        base_url = f'http{"s" if "learning-loop.ai" in host else ""}://{host}/api'
        base: str = base_url
        o = environment_reader.organization()
        p = environment_reader.project()
        assert o and p, 'Outbox needs an organization and a projekct '
        self.target_uri = f'{base}/{o}/projects/{p}/images'
        self.log.info(f'Outbox initialized with target_uri: {self.target_uri}')

        self.shutdown_event = None
        self.upload_thread = None

    def save(self, image: bytes, detections: Detections = Detections(), tags: List[str] = []) -> None:
        id = datetime.now().isoformat(sep='_', timespec='milliseconds')
        tmp = f'{GLOBALS.data_folder}/tmp/{id}'
        detections.tags = tags
        detections.date = id
        os.makedirs(tmp, exist_ok=True)
        with open(tmp + '/image.json', 'w') as f:
            json.dump(jsonable_encoder(detections), f)

        with open(tmp + '/image.jpg', 'wb') as f:
            f.write(image)
        os.rename(tmp, self.path + '/' + id)  # NOTE rename is atomic so upload can run in parallel

    def get_data_files(self):
        return glob(f'{self.path}/*')

    def start_continuous_upload(self):
        self.shutdown_event = Event()
        self.upload_thread = Thread(target=self._continuous_upload)
        self.upload_thread.start()

    def _continuous_upload(self):
        self.log.info('start continuous upload')
        while not self.shutdown_event.is_set():
            self.upload()
            time.sleep(1)
        self.log.info('stop continuous upload')

    def upload(self):
        items = self.get_data_files()
        if items:
            self.log.info(f'Found {len(items)} images to upload')
        for item in items:
            if self.shutdown_event and self.shutdown_event.is_set():
                break
            try:
                data = [('files', open(f'{item}/image.json', 'r')),
                        ('files', open(f'{item}/image.jpg', 'rb'))]

                response = requests.post(self.target_uri, files=data)
                if response.status_code == 200:
                    shutil.rmtree(item)
                    self.log.info(f'uploaded {item} successfully')
                elif response.status_code == 422:
                    self.log.error(f'Broken content in {item}: dropping this data')
                    shutil.rmtree(item)
                else:
                    self.log.error(f'Could not upload {item}: {response.status_code}, {response.content}')
            except:
                self.log.exception('could not upload files')

    def stop_continuous_upload(self, timeout=5):
        proc = None
        try:
            self.shutdown_event.set()
            proc = self.upload_thread
            proc.join(timeout)
        except:
            pass
        if proc and proc.is_alive():
            proc.terminate()
            self.log.info('terminated process')
