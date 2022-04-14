import numpy as np
from learning_loop_node import DetectorNode
import pytest
from conftest import get_outbox_files
import asyncio


@pytest.mark.asyncio
async def test_default_relevance_filter_is_used_by_node(test_detector_node: DetectorNode):
    assert test_detector_node.outbox.path.startswith('/tmp')
    assert len(get_outbox_files(test_detector_node.outbox)) == 0

    image = np.fromfile('detector/tests/test.jpg', np.uint8)
    _ = await test_detector_node.get_detections(image, '00:.....', tags=[])
    # NOTE adding second images with identical detections
    _ = await test_detector_node.get_detections(image, '00:.....', tags=[])
    await asyncio.sleep(1)  # files are stored asynchronously

    assert len(get_outbox_files(test_detector_node.outbox)) == 2,\
        'There should be 1 image and 1 .json file in the outbox'


@pytest.mark.asyncio
async def test_node_can_be_queried_without_any_filter(test_detector_node: DetectorNode):
    assert test_detector_node.outbox.path.startswith('/tmp')
    assert len(get_outbox_files(test_detector_node.outbox)) == 0

    image = np.fromfile('detector/tests/test.jpg', np.uint8)
    _ = await test_detector_node.get_detections(image, '00:.....', tags=[], autoupload='all')
    # NOTE adding second images with identical detections
    _ = await test_detector_node.get_detections(image, '00:.....', tags=[], autoupload='all')
    await asyncio.sleep(1)  # files are stored asynchronously

    assert len(get_outbox_files(test_detector_node.outbox)) == 4,\
        'there should be 2 images and 2 .json file in the outbox'
