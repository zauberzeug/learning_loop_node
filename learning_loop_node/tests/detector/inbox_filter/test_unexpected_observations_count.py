import os
from typing import List

import numpy as np
import pytest

from ....data_classes.image_metadata import BoxDetection, ImageMetadata, PointDetection
from ....detector.inbox_filter.relevance_filter import RelevanceFilter
from ....detector.outbox import Outbox

h_conf_box_det = BoxDetection(category_name='dirt', x=0, y=0, width=100,
                              height=100, category_id='xyz', confidence=.9, model_name='test_model',)
h_conf_point_det = PointDetection(category_name='point', x=100, y=100,
                                  category_id='xyz', confidence=.9, model_name='test_model',)
l_conf_point_det = PointDetection(category_name='point', x=100, y=100,
                                  category_id='xyz', confidence=.3, model_name='test_model',)


@pytest.mark.parametrize(
    "metadata,reason",
    [(ImageMetadata(box_detections=[h_conf_box_det] * 40, point_detections=[h_conf_point_det] * 40),
      ['unexpected_observations_count']),
     (ImageMetadata(box_detections=[h_conf_box_det], point_detections=[h_conf_point_det]), []),
     (ImageMetadata(box_detections=[h_conf_box_det] * 40, point_detections=[l_conf_point_det] * 40),
      ['uncertain', 'unexpected_observations_count']),
     (ImageMetadata(box_detections=[h_conf_box_det], point_detections=[l_conf_point_det]),
      ['uncertain'])])
@pytest.mark.asyncio
async def test_unexpected_observations_count(metadata: ImageMetadata, reason: List[str]):
    os.environ['LOOP_ORGANIZATION'] = 'zauberzeug'
    os.environ['LOOP_PROJECT'] = 'demo'
    outbox = Outbox()

    relevance_filter = RelevanceFilter(outbox)
    assert await relevance_filter.may_upload_detections(metadata, image=np.zeros((300, 600, 3), dtype=np.uint8), cam_id='0:0:0:0') == reason
