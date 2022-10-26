import os
from glob import glob
from fastapi.encoders import jsonable_encoder
import os
from learning_loop_node.detector import Detections
from typing import List
import json
from datetime import datetime
from ...globals import GLOBALS
from ... import environment_reader
from multiprocessing import Event
from .upload_process import UploadProcess


class Outbox():

    def __init__(self) -> None:
        self.path = f'{GLOBALS.data_folder}/outbox'
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        host = os.environ.get('HOST', 'learning-loop.ai')
        base: str = f'http{"s" if host != "backend" else ""}://' + host
        o = environment_reader.organization()
        p = environment_reader.project()
        assert o and p, 'Outbox needs an organization and a projekct '
        self.target_uri = f'{base}/api/{o}/projects/{p}/images'

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

    def create_upload_process(self, shutdown: Event, **kwargs) -> UploadProcess:
        return UploadProcess(shutdown, self.target_uri, self.path, **kwargs)

    def get_data_files(self):
        return glob(f'{self.path}/*')
