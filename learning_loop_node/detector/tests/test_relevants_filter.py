import numpy as np
from learning_loop_node import DetectorNode
import pytest
from conftest import get_outbox_files
import asyncio


@pytest.mark.asyncio
async def test_relevants_filter_is_used_by_node(test_detector_node: DetectorNode):
    assert test_detector_node.outbox.path.startswith('/tmp')
    assert len(get_outbox_files(test_detector_node.outbox)) == 0

    image = np.fromfile('detector/tests/test.jpg', np.uint8)
    _detections = await test_detector_node.get_detections(image, '00:.....', [])
    await asyncio.sleep(0.1)  # files are stored asynchronously

    assert len(get_outbox_files(test_detector_node.outbox)) == 2, 'There should be 1 image and 1 .json file'
