import asyncio
import os

import numpy as np
import pytest

from ...data_classes import BoxDetection, Detections, PointDetection
from ...detector.detector_node import DetectorNode
from .conftest import get_outbox_files
from .testing_detector import TestingDetectorLogic

file_path = os.path.abspath(__file__)
test_image_path = os.path.join(os.path.dirname(file_path), 'test.jpg')


@pytest.mark.parametrize('autoupload, expected_file_count', [(None, 2), ('all', 4)])
async def test_filter_is_used_by_node(test_detector_node: DetectorNode, autoupload, expected_file_count):
    """Test if filtering is used by the node. In particular, when upload is filtered, the identical detections should not be uploaded twice.
    Note thatt we have to mock the dummy detections to only return a point and a box detection."""

    assert isinstance(test_detector_node.detector_logic, TestingDetectorLogic)
    test_detector_node.detector_logic.det_to_return = Detections(
        box_detections=[
            BoxDetection(category_name='some_category_name', x=1, y=2, height=3, width=4,
                         model_name='some_model', confidence=.42, category_id='some_id')],
        point_detections=[
            PointDetection(category_name='some_category_name_2', x=10, y=12,
                           model_name='some_model', confidence=.42, category_id='some_id_2')],)

    assert test_detector_node.outbox.path.startswith('/tmp')
    assert len(get_outbox_files(test_detector_node.outbox)) == 0

    image = np.fromfile(file=test_image_path, dtype=np.uint8)
    _ = await test_detector_node.get_detections(image, '00:.....', tags=[], autoupload=autoupload)
    # NOTE adding second images with identical detections
    _ = await test_detector_node.get_detections(image, '00:.....', tags=[], autoupload=autoupload)
    await asyncio.sleep(.5)  # files are stored asynchronously

    assert len(get_outbox_files(test_detector_node.outbox)) == expected_file_count, \
        'There should be 1 image and 1 .json file for every detection in the outbox'
