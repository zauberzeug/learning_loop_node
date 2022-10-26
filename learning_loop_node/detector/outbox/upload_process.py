from multiprocessing import Process, Event
from glob import glob
import requests
import shutil
import logging
import time


class UploadProcess(Process):
    def __init__(self, shutdown: Event, target_uri: str, data_path: str, **kwargs):
        self.shutdown = shutdown
        self.target_uri = target_uri
        self.data_path = data_path
        self.log = logging.getLogger()
        super().__init__(**kwargs)

    def run(self):
        self.log.info('Uploading process started.')
        while True:
            if self.shutdown.is_set():
                break
            self.upload()
            time.sleep(1)

    def upload(self):
        items = self.get_data_files()
        self.log.info(f'Found {len(items)} images to upload')
        for item in items:
            if self.shutdown.is_set():
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
                self.log.s.exception('could not upload files')

    def get_data_files(self):
        return glob(f'{self.data_path}/*')
