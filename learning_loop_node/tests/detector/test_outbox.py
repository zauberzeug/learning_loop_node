import asyncio
import io
import os
import shutil

import pytest
from PIL import Image

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


@pytest.fixture(autouse=True, scope='session')
async def fix_upload_bug():
    """ This is a workaround for an upload bug that causes the SECOND upload to fail on the CI server. """
    os.environ['LOOP_ORGANIZATION'] = 'zauberzeug'
    os.environ['LOOP_PROJECT'] = 'demo'
    shutil.rmtree(f'{GLOBALS.data_folder}/outbox', ignore_errors=True)
    test_outbox = Outbox()

    await test_outbox.set_mode('continuous_upload')
    await test_outbox.save(get_test_image_binary())
    await asyncio.sleep(6)
    assert await wait_for_outbox_count(test_outbox, 0, timeout=15), 'File was not cleared even though outbox should be in continuous_upload'
    assert test_outbox.upload_counter == 1

    await test_outbox.save(get_test_image_binary())
    await asyncio.sleep(6)
    # assert await wait_for_outbox_count(test_outbox, 0, timeout=90), 'File was not cleared even though outbox should be in continuous_upload'
    # assert test_outbox.upload_counter == 2

    await test_outbox.set_mode('stopped')
    shutil.rmtree(test_outbox.path, ignore_errors=True)


@pytest.mark.asyncio
async def test_set_outbox_mode(test_outbox: Outbox):
    await test_outbox.set_mode('stopped')
    await test_outbox.save(get_test_image_binary())
    assert await wait_for_outbox_count(test_outbox, 1)
    await asyncio.sleep(6)
    assert await wait_for_outbox_count(test_outbox, 1), 'File was cleared even though outbox should be stopped'

    await test_outbox.set_mode('continuous_upload')
    assert await wait_for_outbox_count(test_outbox, 0, timeout=15), 'File was not cleared even though outbox should be in continuous_upload'
    assert test_outbox.upload_counter == 1


@pytest.mark.asyncio
async def test_invalid_jpg_is_not_saved(test_outbox: Outbox):
    invalid_bytes = b'invalid jpg'
    await test_outbox.save(invalid_bytes)
    assert len(test_outbox.get_upload_folders()) == 0


# ------------------------------ Helper functions --------------------------------------


def get_test_image_binary() -> bytes:
    img = Image.new('RGB', (600, 300), color=(73, 109, 137))
    # convert img to jpg binary

    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

    # return img.tobytes() # NOT WORKING

    # img.save('/tmp/image.jpg')
    # with open('/tmp/image.jpg', 'rb') as f:
    #     data = f.read()

    # return data

    # img = np.ones((300, 300, 3), np.uint8)*255 # NOT WORKING
    # return img.tobytes()


async def wait_for_outbox_count(outbox: Outbox, count: int, timeout: int = 10) -> bool:
    for _ in range(timeout):
        if len(outbox.get_upload_folders()) == count:
            return True
        await asyncio.sleep(1)
    return False
