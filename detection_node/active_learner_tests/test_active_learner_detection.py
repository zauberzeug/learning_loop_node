import pytest
from active_learner import detection
import time


def test_aging():
    _detection = detection.ActiveLearnerDetection(None, None, None, None, None, None, None)
    time.sleep(0.2)
    assert _detection._is_older_than(0.1) == True
    assert _detection._is_older_than(0.3) == False


def test_calculate_iou():
    detection_one = detection.ActiveLearnerDetection('dirt', 10, 0, 30, 100, 'a', 30)
    detection_two = detection.ActiveLearnerDetection('dirt', 20, 0, 30, 100, 'b', 61)
    detection_three = detection.ActiveLearnerDetection('dirt', 0, 30, 10, 10, 'b', 61)

    assert detection_one.intersection_over_union(detection_two) == 0.5
    assert detection_one.intersection_over_union(detection_three) == 0.0
