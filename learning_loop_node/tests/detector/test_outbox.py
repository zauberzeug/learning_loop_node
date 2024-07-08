import asyncio
import io
import os
import shutil

import pytest
from PIL import Image

from ...data_classes import Detections
from ...detector.detector_node import DetectorNode
from ...detector.outbox import Outbox
from ...globals import GLOBALS

# pylint: disable=redefined-outer-name


@pytest.fixture()
async def test_outbox():
    os.environ['LOOP_ORGANIZATION'] = 'zauberzeug'
    os.environ['LOOP_PROJECT'] = 'demo'
    shutil.rmtree(f'{GLOBALS.data_folder}/outbox', ignore_errors=True)
    test_outbox = Outbox()

    yield test_outbox
    await test_outbox.set_mode('stopped')
    shutil.rmtree(test_outbox.path, ignore_errors=True)


@pytest.mark.asyncio
async def test_files_are_automatically_uploaded_by_node(test_detector_node: DetectorNode):
    test_detector_node.outbox.save(get_test_image_binary(), Detections())
    assert await wait_for_outbox_count(test_detector_node.outbox, 1)
    assert await wait_for_outbox_count(test_detector_node.outbox, 0)


@pytest.mark.asyncio
async def test_set_outbox_mode(test_outbox: Outbox):
    await test_outbox.set_mode('stopped')
    test_outbox.save(get_test_image_binary())
    assert await wait_for_outbox_count(test_outbox, 1)
    await asyncio.sleep(6)
    assert await wait_for_outbox_count(test_outbox, 1), 'File was cleared even though outbox should be stopped'

    await test_outbox.set_mode('continuous_upload')
    assert await wait_for_outbox_count(test_outbox, 0), 'File was not cleared even though outbox should be in continuous_upload'
    assert test_outbox.upload_counter == 1


@pytest.mark.asyncio
async def test_outbox_upload_is_successful(test_outbox: Outbox):
    test_outbox.save(get_test_image_binary())
    await asyncio.sleep(1)
    test_outbox.save(get_test_image_binary())
    assert await wait_for_outbox_count(test_outbox, 2)
    test_outbox.upload()
    assert await wait_for_outbox_count(test_outbox, 0)
    assert test_outbox.upload_counter == 2


@pytest.mark.asyncio
async def test_invalid_jpg_is_not_saved(test_outbox: Outbox):
    invalid_bytes = b'invalid jpg'
    test_outbox.save(invalid_bytes)
    assert len(test_outbox.get_data_files()) == 0


# ------------------------------ Helper functions --------------------------------------


def get_test_image_binary():
    img = Image.new('RGB', (60, 30), color=(73, 109, 137))
    # convert img to jpg binary

    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

    # return img.tobytes() # NOT WORKING

    # img.save('/tmp/image.jpg')
    # with open('/tmp/image.jpg', 'rb') as f:
    #     data = f.read()

    # return data

    # img = np.ones((300, 300, 3), np.uint8)*255 # NOT WORKING
    # return img.tobytes()


async def wait_for_outbox_count(outbox: Outbox, count: int, timeout: int = 10) -> bool:
    for _ in range(timeout):
        if len(outbox.get_data_files()) == count:
            return True
        await asyncio.sleep(1)
    return False
