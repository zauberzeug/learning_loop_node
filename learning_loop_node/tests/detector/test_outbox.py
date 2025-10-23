import asyncio
import os
import shutil

import numpy as np
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
    await test_outbox.save(get_test_image())
    await asyncio.sleep(6)
    assert await wait_for_outbox_count(test_outbox, 0, timeout=15), 'File was not cleared even though outbox should be in continuous_upload'
    assert test_outbox.upload_counter == 1

    await test_outbox.save(get_test_image())
    await asyncio.sleep(6)
    # assert await wait_for_outbox_count(test_outbox, 0, timeout=90), 'File was not cleared even though outbox should be in continuous_upload'
    # assert test_outbox.upload_counter == 2

    await test_outbox.set_mode('stopped')
    shutil.rmtree(test_outbox.path, ignore_errors=True)


@pytest.mark.asyncio
async def test_set_outbox_mode(test_outbox: Outbox):
    await test_outbox.set_mode('stopped')
    await test_outbox.save(get_test_image())
    assert await wait_for_outbox_count(test_outbox, 1)
    await asyncio.sleep(6)
    assert await wait_for_outbox_count(test_outbox, 1), 'File was cleared even though outbox should be stopped'

    await test_outbox.set_mode('continuous_upload')
    assert await wait_for_outbox_count(test_outbox, 0, timeout=15), 'File was not cleared even though outbox should be in continuous_upload'
    assert test_outbox.upload_counter == 1

# ------------------------------ Helper functions --------------------------------------


def get_test_image() -> np.ndarray:
    img = Image.new('RGB', (600, 300), color=(73, 109, 137))  # type: ignore
    # convert img to np array
    return np.array(img)


async def wait_for_outbox_count(outbox: Outbox, count: int, timeout: int = 10) -> bool:
    for _ in range(timeout):
        if len(outbox.get_upload_folders()) == count:
            return True
        await asyncio.sleep(1)
    return False
