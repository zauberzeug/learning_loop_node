from learning_loop_node.tests import test_helper
from learning_loop_node.gdrive_downloader import g_download
from learning_loop_node.rest import downloads
from learning_loop_node.context import Context
import pytest
import logging
import os
from icecream import ic
from learning_loop_node.globals import GLOBALS


@pytest.mark.skip  # NOTE: Brauchen wir diesen Test noch?
async def test_multiple_get_requests_after_post_request_should_not_causes_timeout_error():
    logging.debug('downloading model from gdrive')

    # folder: https://drive.google.com/drive/u/0/folders/1c1oxQwhj-bjqbYjM0B2q7I9ZZe-s2SWD
    # filename: zuckerruebe_roboter_1.5_yolov5_pytorch.zip
    file_id = '1sZWa053fWT9PodrujDX90psmjhFVLyBV'
    destination = '/tmp/model.zip'
    g_download(file_id, destination)
    test_helper.unzip(destination, '/tmp/some_model')

    data = test_helper.prepare_formdata(['/tmp/some_model/model.pt', '/tmp/some_model/model.json'])
    from learning_loop_node.loop import loop
    async with loop.post(f'api/zauberzeug/projects/pytest/models/yolov5_pytorch', data=data) as response:
        if response.status != 200:
            msg = f'unexpected status code {response.status} while putting model'
            logging.error(msg)
            raise (Exception(msg))
        model = await response.json()

    for i in range(10):
        ic('going to download model from loop')

        import shutil
        shutil.rmtree(GLOBALS.data_folder, ignore_errors=True)
        os.makedirs(GLOBALS.data_folder)
        await downloads.download_model(GLOBALS.data_folder, Context(organization='zauberzeug', project='pytest'), model['id'], 'yolov5_pytorch')
