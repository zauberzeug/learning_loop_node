import pytest
from detection import Detection
from active_learner import watcher


def test_calculate_iou():
    detection_one = Detection('dirt', 10, 0, 30, 100, 'a', 30)
    detection_two = Detection('dirt', 20, 0, 30, 100, 'b', 61)

    detection_to_watch = watcher.Watcher(detection_one)
    assert detection_to_watch.intersection_over_union(detection_two) == 0.5
