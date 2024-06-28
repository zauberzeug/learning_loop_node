import time
from datetime import datetime, timedelta

from ....data_classes import BoxDetection, Observation


def test_aging():
    _detection = Observation(BoxDetection(category_name='dirt', x=10, y=0, height=30, width=100,
                             model_name='model', confidence=0.30, category_id='some-id'))
    time.sleep(0.2)
    assert _detection.is_older_than(0.1)
    assert not _detection.is_older_than(0.3)


def test_calculate_iou():
    one = BoxDetection(category_name='dirt', x=10, y=0, width=30, height=100,
                       model_name='model', confidence=0.30, category_id='some-id')
    two = BoxDetection(category_name='dirt', x=20, y=0, width=30, height=100,
                       model_name='model', confidence=0.61, category_id='some-id')
    three = BoxDetection(category_name='dirt', x=0, y=30, width=10, height=10,
                         model_name='model', confidence=0.61, category_id='some-id')

    assert one.intersection_over_union(two) == 0.5
    assert one.intersection_over_union(three) == 0.0


def test_update_last_seen():
    observation = Observation(BoxDetection(category_name='dirt', x=10, y=0, height=30,
                              width=100, model_name='model', confidence=0.30, category_id='some-id'))
    observation.last_seen = datetime.now() - timedelta(seconds=.5)
    assert observation.is_older_than(0.5)
