import os
from typing import List

import pytest

from learning_loop_node.data_classes.detections import (BoxDetection,
                                                        Detections,
                                                        PointDetection)
from learning_loop_node.detector.outbox import Outbox
from learning_loop_node.inbox_filter.relevance_filter import RelevanceFilter

high_conf_box_detection = BoxDetection(category_name='dirt', x=0, y=0, width=100,
                                       height=100, category_id='xyz', confidence=.9, model_name='test_model',)
high_conf_point_detection = PointDetection(category_name='point', x=100, y=100,
                                           category_id='xyz', confidence=.9, model_name='test_model',)
low_conf_point_detection = PointDetection(category_name='point', x=100, y=100,
                                          category_id='xyz', confidence=.3, model_name='test_model',)


@pytest.mark.parametrize(
    "detections,reason",
    [(Detections(box_detections=[high_conf_box_detection] * 40, point_detections=[high_conf_point_detection] * 40),
      ['unexpected_observations_count']),
     (Detections(box_detections=[high_conf_box_detection],
                 point_detections=[high_conf_point_detection]),
      []),
     (Detections(box_detections=[high_conf_box_detection] * 40, point_detections=[low_conf_point_detection] * 40),
      ['uncertain', 'unexpected_observations_count']),
     (Detections(box_detections=[high_conf_box_detection],
                 point_detections=[low_conf_point_detection]),
      ['uncertain'])])
def test_unexpected_observations_count(detections: Detections, reason: List[str]):

    os.environ['LOOP_ORGANIZATION'] = 'zauberzeug'
    os.environ['LOOP_PROJECT'] = 'demo'
    outbox = Outbox()

    r_filter = RelevanceFilter(outbox)
    assert r_filter.learn(detections, raw_image=b'', camera_id='0:0:0:0', tags=[]) == reason
