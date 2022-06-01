# group Tests incoming
from datetime import datetime, timedelta
from learning_loop_node.detector.box_detection import BoxDetection
from learning_loop_node.detector.point_detection import PointDetection
from learning_loop_node.detector.segmentation_detection import Point, SegmentationDetection, Shape
from learning_loop_node.inbox_filter.relevance_group import RelevanceGroup

dirt_detection = BoxDetection('dirt', 0, 0, 100, 100, 'xyz', .3)
second_dirt_detection = BoxDetection('dirt', 0, 20, 10, 10, 'xyz', .35)
conf_too_high_detection = BoxDetection('dirt', 0, 0, 100, 100, 'xyz', .61)
conf_too_low_detection = BoxDetection('dirt', 0, 0, 100, 100, 'xyz', .29)


def test_group_confidence():
    group = RelevanceGroup()
    assert len(group.recent_observations) == 0

    filter_cause = group.add_box_detections([dirt_detection])
    assert filter_cause == ['uncertain'], f'Active Learning should be done due to uncertain'
    assert len(group.recent_observations) == 1, 'Detection should be stored'

    filter_cause = group.add_box_detections([dirt_detection])
    assert len(group.recent_observations) == 1, f'Detection should already be stored'
    assert filter_cause == []

    filter_cause = group.add_box_detections([conf_too_low_detection])
    assert len(group.recent_observations) == 1, 'Confidence of detection too low'
    assert filter_cause == []

    filter_cause = group.add_box_detections([conf_too_high_detection])
    assert len(group.recent_observations) == 1, 'Confidence of detection too high'
    assert filter_cause == []


def test_add_second_detection_to_group():
    group = RelevanceGroup()
    assert len(group.recent_observations) == 0
    group.add_box_detections([dirt_detection])
    assert len(group.recent_observations) == 1, 'Detection should be stored'
    group.add_box_detections([second_dirt_detection])
    assert len(group.recent_observations) == 2, 'Second detection should be stored'


def test_forget_old_detections():
    group = RelevanceGroup()
    assert len(group.recent_observations) == 0

    filter_cause = group.add_box_detections([dirt_detection])
    assert filter_cause == ['uncertain'], 'Active Learning should be done due to uncertain.'

    assert len(group.recent_observations) == 1

    group.recent_observations[0].last_seen = datetime.now() - timedelta(minutes=30)
    group.forget_old_detections()
    assert len(group.recent_observations) == 1

    group.recent_observations[0].last_seen = datetime.now() - timedelta(hours=1, minutes=1)
    group.forget_old_detections()
    assert len(group.recent_observations) == 0


def test_active_group_extracts_from_json():
    detections = [
        {"category_name": "dirt",
         "x": 1366,
         "y": 1017,
         "width": 37,
         "height": 24,
         "model_name": "some_weightfile",
         "confidence": .3},
        {"category_name": "obstacle",
         "x": 0,
         "y": 0,
         "width": 37,
         "height": 24,
         "model_name": "some_weightfile",
         "confidence": .35},
        {"category_name": "dirt",
         "x": 1479,
         "y": 862,
         "width": 14,
         "height": 11,
         "model_name": "some_weightfile",
         "confidence": .2}]

    camera_id = '0000'
    groups = {camera_id: RelevanceGroup()}

    filter_cause = groups[camera_id].add_box_detections(
        [BoxDetection.from_dict(_detection) for _detection in detections]
    )

    assert filter_cause == ['uncertain']


def test_segmentation_detections_are_extracted_from_json():
    seg_detection = {"category_name": "seg",
                     "shape": [Point(x=193, y=876), Point(x=602, y=193), Point(x=121, y=8)],
                     "model_name": "some_weightfile",
                     "confidence": .3}

    camera_id = '0000'
    groups = {camera_id: RelevanceGroup()}

    filter_cause = groups[camera_id].add_segmentation_detections(
        [SegmentationDetection.from_dict(seg_detection)]
    )
    assert filter_cause == ['segmentation_detection']


def test_ignoring_similar_points():
    group = RelevanceGroup()
    filter_cause = group.add_point_detections(
        [PointDetection('point', 100, 100, 'xyz', 0.3)]
    )
    assert filter_cause == ['uncertain'], 'Active Learning should be done due to low confidence'
    assert len(group.recent_observations) == 1, 'detection should be stored'

    filter_cause = group.add_point_detections(
        [PointDetection('point', 104, 98, 'xyz', 0.3)])
    assert len(group.recent_observations) == 1, f'detection should already be stored'
    assert filter_cause == []


def test_getting_low_confidence_points():
    group = RelevanceGroup()
    filter_cause = group.add_point_detections(
        [PointDetection('point', 100, 100, 'xyz', 0.3)],
    )
    assert filter_cause == ['uncertain'], 'Active Learning should be done due to low confidence'
    assert len(group.recent_observations) == 1, 'detection should be stored'

    filter_cause = group.add_point_detections([PointDetection('point', 104, 98, 'xyz', 0.3)])
    assert len(group.recent_observations) == 1, 'detection should already be stored'
    assert filter_cause == []


def test_getting_segmentation_detections():
    group = RelevanceGroup()
    filter_cause = group.add_segmentation_detections(
        [SegmentationDetection('segmentation', Shape([Point(100, 200), Point(300, 400)]), 'xyz', 0.3)],
    )
    assert filter_cause == ['segmentation_detection'], 'all segmentation detections are collected'
    assert len(group.recent_observations) == 1, 'detection should be stored'

    filter_cause = group.add_segmentation_detections(
        [SegmentationDetection('segmentation', Shape([Point(105, 205), Point(305, 405)]), 'xyz', 0.3)],
    )
    assert len(group.recent_observations) == 2, 'segmentation detections are not filtered by similarity'
    assert filter_cause == ['segmentation_detection']
