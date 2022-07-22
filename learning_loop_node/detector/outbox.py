import os
from glob import glob
from fastapi.encoders import jsonable_encoder
import os
from learning_loop_node.detector import Detections
from typing import List
import json
from datetime import datetime
import requests
import logging
import shutil
from ..globals import GLOBALS


class Outbox():

    def __init__(self) -> None:
        self.path = f'{GLOBALS.data_folder}/outbox'
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        host = os.environ.get('HOST', 'learning-loop.ai')
        base: str = f'http{"s" if host != "backend" else ""}://' + host
        o = os.environ.get('ORGANIZATION')
        p = os.environ.get('PROJECT')
        self.target_uri = f'{base}/api/{o}/projects/{p}/images'
        self.upload_in_progress = False

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

    def upload(self) -> None:
        if self.upload_in_progress:
            return
        self.upload_in_progress = True

        try:
            for item in self.get_data_files():
                data = [('files', open(f'{item}/image.json', 'r')),
                        ('files', open(f'{item}/image.jpg', 'rb'))]

                response = requests.post(self.target_uri, files=data)
                if response.status_code == 200:
                    shutil.rmtree(item)
                    logging.info(f'uploaded {item} successfully')
                elif response.status_code == 422:
                    logging.error(f'Broken content in {item}: dropping this data')
                    shutil.rmtree(item)
                else:
                    logging.error(f'Could not upload {item}: {response.status_code}, {response.content}')
        except:
            logging.exception('could not upload files')
        self.upload_in_progress = False
