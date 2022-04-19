from datetime import datetime, timedelta
from learning_loop_node.inbox_filter.observation import Observation
from learning_loop_node.detector.box_detection import BoxDetection
import time


def test_aging():
    _detection = Observation(BoxDetection(None, None, None, None, None, None, None))
    time.sleep(0.2)
    assert _detection.is_older_than(0.1) == True
    assert _detection.is_older_than(0.3) == False


def test_calculate_iou():
    one = BoxDetection('dirt', 10, 0, 30, 100, None, 30)
    two = BoxDetection('dirt', 20, 0, 30, 100, None, 61)
    three = BoxDetection('dirt', 0, 30, 10, 10, None, 61)

    assert one.intersection_over_union(two) == 0.5
    assert one.intersection_over_union(three) == 0.0


def test_update_last_seen():
    observation = Observation(BoxDetection(None, None, None, None, None, None, None))
    observation.last_seen = datetime.now() - timedelta(seconds=.5)
    assert observation.is_older_than(0.5) == True
