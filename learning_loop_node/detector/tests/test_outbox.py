import os
import shutil

import numpy as np
import pytest
from PIL import Image

from learning_loop_node.data_classes import Detections
from learning_loop_node.detector.detector_node import DetectorNode
from learning_loop_node.detector.outbox import Outbox


@pytest.fixture()
def test_outbox():
    os.environ['LOOP_ORGANIZATION'] = 'zauberzeug'
    os.environ['LOOP_PROJECT'] = 'demo'
    test_outbox = Outbox()
    shutil.rmtree(test_outbox.path, ignore_errors=True)
    os.mkdir(test_outbox.path)

    yield test_outbox
    shutil.rmtree(test_outbox.path, ignore_errors=True)


def test_files_are_deleted_after_sending(test_outbox: Outbox):
    assert test_outbox.path.startswith('/tmp')
    os.mkdir(f'{test_outbox.path}/test')
    with open(f'{test_outbox.path}/test/image.json', 'w') as f:
        f.write('{"box_detections":[]}')
        f.close()

    img = Image.new('RGB', (60, 30), color=(73, 109, 137))
    img.save(f'{test_outbox.path}/test/image.jpg')

    items = test_outbox.get_data_files()
    assert len(items) == 1

    test_outbox.upload()

    items = test_outbox.get_data_files()
    assert len(items) == 0


def test_saving_opencv_image(test_outbox: Outbox):
    img = np.ones((300, 300, 1), np.uint8)*255
    test_outbox.save(img.tobytes())
    items = test_outbox.get_data_files()
    assert len(items) == 1


def test_saving_binary(test_outbox: Outbox):
    assert len(test_outbox.get_data_files()) == 0
    img = Image.new('RGB', (60, 30), color=(73, 109, 137))
    img.save('/tmp/image.jpg')
    with open('/tmp/image.jpg', 'rb') as f:
        data = f.read()
    test_outbox.save(data)
    assert len(test_outbox.get_data_files()) == 1


@pytest.mark.asyncio
async def test_files_are_automatically_uploaded(test_detector_node: DetectorNode):
    test_detector_node.outbox.save(Image.new('RGB', (60, 30), color=(73, 109, 137)).tobytes(), Detections())
    assert len(test_detector_node.outbox.get_data_files()) == 1

    assert len(test_detector_node.outbox.get_data_files()) == 1
