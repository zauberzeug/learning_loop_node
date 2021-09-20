from glob import glob
import os
from ..loop import loop
from urllib.parse import urljoin
from requests import Session
import asyncio


class LiveServerSession(Session):
    """https://stackoverflow.com/a/51026159/364388"""

    def __init__(self, *args, **kwargs):
        super(LiveServerSession, self).__init__(*args, **kwargs)
        self.prefix_url = loop.base_url

    def request(self, method, url, *args, **kwargs):
        url = urljoin(self.prefix_url, url)
        headers = {}
        if 'token' in url:
            return super(LiveServerSession, self).request(method, url, *args, **kwargs)

        headers = loop.get_headers()
        return super(LiveServerSession, self).request(method, url, headers=headers, *args, **kwargs)


def get_files_in_folder(folder: str):
    files = [entry for entry in glob(f'{folder}/**/*', recursive=True) if os.path.isfile(entry)]
    files.sort()
    return files
