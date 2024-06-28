import os
from typing import List

import pytest

from ....data_classes.detections import BoxDetection, Detections, PointDetection
from ....detector.inbox_filter.relevance_filter import RelevanceFilter
from ....detector.outbox import Outbox

h_conf_box_det = BoxDetection(category_name='dirt', x=0, y=0, width=100,
                              height=100, category_id='xyz', confidence=.9, model_name='test_model',)
h_conf_point_det = PointDetection(category_name='point', x=100, y=100,
                                  category_id='xyz', confidence=.9, model_name='test_model',)
l_conf_point_det = PointDetection(category_name='point', x=100, y=100,
                                  category_id='xyz', confidence=.3, model_name='test_model',)


@pytest.mark.parametrize(
    "detections,reason",
    [(Detections(box_detections=[h_conf_box_det] * 40, point_detections=[h_conf_point_det] * 40),
      ['unexpected_observations_count']),
     (Detections(box_detections=[h_conf_box_det], point_detections=[h_conf_point_det]), []),
     (Detections(box_detections=[h_conf_box_det] * 40, point_detections=[l_conf_point_det] * 40),
      ['uncertain', 'unexpected_observations_count']),
     (Detections(box_detections=[h_conf_box_det], point_detections=[l_conf_point_det]),
      ['uncertain'])])
def test_unexpected_observations_count(detections: Detections, reason: List[str]):
    os.environ['LOOP_ORGANIZATION'] = 'zauberzeug'
    os.environ['LOOP_PROJECT'] = 'demo'
    outbox = Outbox()

    r_filter = RelevanceFilter(outbox)
    assert r_filter.may_upload_detections(detections, raw_image=b'', cam_id='0:0:0:0', tags=[]) == reason
