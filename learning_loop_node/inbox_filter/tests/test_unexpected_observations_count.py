from typing import List

import pytest

from learning_loop_node.detector.detections import (BoxDetection, Detections,
                                                    PointDetection)
from learning_loop_node.detector.outbox.outbox import Outbox
from learning_loop_node.inbox_filter.relevance_filter import RelevanceFilter

high_conf_box_detection = BoxDetection('dirt', 0, 0, 100, 100, 'xyz', .9)
high_conf_point_detection = PointDetection('point', 100, 100, 'xyz', .9)
low_conf_point_detection = PointDetection('point', 100, 100, 'xyz', .3)


@pytest.mark.parametrize(
    "detections,reason",
    [(Detections(box_detections=[high_conf_box_detection] * 40, point_detections=[high_conf_point_detection] * 40),
      ['unexpectedObservationsCount']),
     (Detections(box_detections=[high_conf_box_detection],
                 point_detections=[high_conf_point_detection]),
      []),
     (Detections(box_detections=[high_conf_box_detection] * 40, point_detections=[low_conf_point_detection] * 40),
      ['uncertain', 'unexpectedObservationsCount']),
     (Detections(box_detections=[high_conf_box_detection],
                 point_detections=[low_conf_point_detection]),
      ['uncertain'])])
def test_unexpected_observations_count(detections: Detections, reason: List[str]):
    filter = RelevanceFilter(Outbox())
    assert filter.learn(detections, raw_image=b'', camera_id='0:0:0:0', tags=[]) == reason
