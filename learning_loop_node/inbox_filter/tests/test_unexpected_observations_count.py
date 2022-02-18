from typing import List
from learning_loop_node.detector.box_detection import BoxDetection
from learning_loop_node.detector.detections import Detections
from learning_loop_node.detector.outbox import Outbox
from learning_loop_node.detector.point_detection import PointDetection
from learning_loop_node.inbox_filter.relevants_filter import RelevantsFilter
import pytest


high_conf_box_detection = BoxDetection('dirt', 0, 0, 100, 100, 'xyz', .9)
high_conf_point_detection =  PointDetection('point', 100, 100, 'xyz', .9)
low_conf_point_detection = PointDetection('point', 100, 100, 'xyz', .3)

@pytest.mark.parametrize("detections,reason", 
                        [(Detections(box_detections=[high_conf_box_detection]*40, point_detections=[high_conf_point_detection]*40), ['unexpectedObservationsCount']), 
                         (Detections(box_detections=[high_conf_box_detection], point_detections=[high_conf_point_detection]), []),
                         (Detections(box_detections=[high_conf_box_detection]*40, point_detections=[low_conf_point_detection]*40), ['lowConfidence', 'unexpectedObservationsCount']),
                         (Detections(box_detections=[high_conf_box_detection], point_detections=[low_conf_point_detection]), ['lowConfidence'])])
def test_unexpected_observations_count(detections: Detections, reason: List[str]):
    filter = RelevantsFilter(Outbox())
    assert filter._check_detections(detections, '0:0:0:0') == reason