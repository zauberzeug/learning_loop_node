from learning_loop_node.detector.box_detection import BoxDetection
from learning_loop_node.detector.detections import Detections
from learning_loop_node.detector.outbox import Outbox
from learning_loop_node.inbox_filter.relevants_filter import RelevantsFilter


dirt_detection = BoxDetection('dirt', 0, 0, 100, 100, 'xyz', .9)


def test_unexpected_observations_count():
    filter = RelevantsFilter(Outbox())
    assert filter._check_detections(Detections(box_detections=[dirt_detection] * 80), '0:0:0:0') == ['unexpectedObservationsCount']