
from learning_loop_node.detector.detections import Detections
from learning_loop_node.detector.detector_node import DetectorNode
from learning_loop_node.detector.outbox.outbox import Outbox
from PIL import Image
import os
import numpy as np
import pytest
import shutil
from multiprocessing import Event


@pytest.fixture()
def outbox():
    os.environ['LOOP_ORGANIZATION'] = 'zauberzeug'
    os.environ['LOOP_PROJECT'] = 'demo'
    outbox = Outbox()
    shutil.rmtree(outbox.path, ignore_errors=True)
    os.mkdir(outbox.path)

    yield outbox
    shutil.rmtree(outbox.path, ignore_errors=True)


def test_files_are_deleted_after_sending(outbox: Outbox):
    assert outbox.path.startswith('/tmp')
    os.mkdir(f'{outbox.path}/test')
    with open(f'{outbox.path}/test/image.json', 'w') as f:
        f.write('{"box_detections":[]}')
        f.close()

    img = Image.new('RGB', (60, 30), color=(73, 109, 137))
    img.save(f'{outbox.path}/test/image.jpg')

    items = outbox.get_data_files()
    assert len(items) == 1

    outbox.upload()

    items = outbox.get_data_files()
    assert len(items) == 0


def test_saving_opencv_image(outbox: Outbox):
    img = np.ones((300, 300, 1), np.uint8)*255
    outbox.save(img)
    items = outbox.get_data_files()
    assert len(items) == 1


def test_saving_binary(outbox: Outbox):
    assert len(outbox.get_data_files()) == 0
    img = Image.new('RGB', (60, 30), color=(73, 109, 137))
    img.save('/tmp/image.jpg')
    with open(f'/tmp/image.jpg', 'rb') as f:
        data = f.read()
    outbox.save(data)
    assert len(outbox.get_data_files()) == 1


def test_files_are_automatically_uploaded(test_detector_node: DetectorNode):
    test_detector_node.outbox.save(Image.new('RGB', (60, 30), color=(73, 109, 137)).tobytes(), Detections())
    assert len(test_detector_node.outbox.get_data_files()) == 1

    assert len(test_detector_node.outbox.get_data_files()) == 1
