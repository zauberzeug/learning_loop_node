from detection import Detection
import pytest
from active_learner import detection
import time


def test_aging():
    _detection = detection.ActiveLearnerDetection(Detection(None, None, None, None, None, None, None))
    time.sleep(0.2)
    assert _detection._is_older_than(0.1) == True
    assert _detection._is_older_than(0.3) == False


def test_calculate_iou():
    detection_one = detection.ActiveLearnerDetection(Detection('dirt', 10, 0, 30, 100, None, 30))
    detection_two = detection.ActiveLearnerDetection(Detection('dirt', 20, 0, 30, 100, None, 61))
    detection_three = detection.ActiveLearnerDetection(Detection('dirt', 0, 30, 10, 10, None, 61))

    assert detection_one.intersection_over_union(detection_two) == 0.5
    assert detection_one.intersection_over_union(detection_three) == 0.0
