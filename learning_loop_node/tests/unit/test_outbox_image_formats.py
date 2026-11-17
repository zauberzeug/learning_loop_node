from glob import glob
from pathlib import Path

import numpy as np
import pytest

from ...detector.outbox import Outbox
from ...globals import GLOBALS


@pytest.fixture
def outbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Outbox:
    monkeypatch.setattr(GLOBALS, 'data_folder', str(tmp_path))
    monkeypatch.setenv('LOOP_ORGANIZATION', 'zauberzeug')
    monkeypatch.setenv('LOOP_PROJECT', 'demo')
    return Outbox()


async def test_jpeg_bytes_are_stored_as_they_are(outbox: Outbox):
    jpeg_bytes = b'\xff\xd8 whatever came in goes out unchanged'
    await outbox.save(jpeg_bytes)
    assert _saved_image(outbox) == jpeg_bytes


async def test_a_pixel_array_is_encoded_as_jpeg(outbox: Outbox):
    await outbox.save(np.zeros((4, 4, 3), dtype=np.uint8))
    assert _saved_image(outbox)[:2] == b'\xff\xd8'  # the JPEG start marker


def _saved_image(outbox: Outbox) -> bytes:
    files = glob(f'{outbox.path}/normal/*/image_*.jpg')
    assert len(files) == 1
    return Path(files[0]).read_bytes()
