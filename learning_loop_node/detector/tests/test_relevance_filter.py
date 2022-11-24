import numpy as np
from learning_loop_node import DetectorNode
import pytest
from learning_loop_node.detector.tests.conftest import get_outbox_files
import asyncio


@pytest.mark.parametrize('autoupload,expected_file_count', [(None, 2), ('all', 4)])
async def test_filter_is_used_by_node(test_detector_node: DetectorNode, autoupload, expected_file_count):
    assert test_detector_node.outbox.path.startswith('/tmp')
    assert len(get_outbox_files(test_detector_node.outbox)) == 0

    image = np.fromfile('detector/tests/test.jpg', np.uint8)
    _ = await test_detector_node.get_detections(image, '00:.....', tags=[], autoupload=autoupload)
    # NOTE adding second images with identical detections
    _ = await test_detector_node.get_detections(image, '00:.....', tags=[], autoupload=autoupload)
    await asyncio.sleep(1)  # files are stored asynchronously

    assert len(get_outbox_files(test_detector_node.outbox)) == expected_file_count,\
        'There should be 1 image and 1 .json file for every detection in the outbox'
