import numpy as np
import pytest

from ...data_classes import Category, ModelInformation
from ...detector.postprocess import (
    Prediction,
    bbox_iou,
    non_max_suppression,
    post_process,
    predictions_from_xyxy,
    to_detections,
    to_image_metadata,
)
from ...enums import CategoryType

BOX = Category(id='uuid-box', name='car', type=CategoryType.Box)
POINT = Category(id='uuid-point', name='weed', type=CategoryType.Point)


def model_information(*categories: Category) -> ModelInformation:
    return ModelInformation(id='model-uuid', host='localhost', organization='zauberzeug',
                            project='pytest', version='1.2', categories=list(categories or (BOX, POINT)))


# ---------------------------------------------------------------- iou and suppression

def test_identical_boxes_have_an_iou_of_one():
    box = np.array([[0, 0, 10, 10]], dtype=np.float32)
    assert bbox_iou(box, box)[0] == pytest.approx(1.0)


def test_disjoint_boxes_have_an_iou_of_zero():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[100, 100, 110, 110]], dtype=np.float32)
    assert bbox_iou(a, b)[0] == pytest.approx(0.0)


def test_overlapping_boxes_of_one_class_are_suppressed_keeping_the_best():
    boxes = np.array([[10, 10, 60, 60], [12, 12, 62, 62]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    kept_boxes, kept_scores, _ = non_max_suppression(
        boxes, scores, np.array([0, 0]), iou_threshold=0.45, origin_h=200, origin_w=200)
    assert len(kept_boxes) == 1
    assert kept_scores[0] == pytest.approx(0.9)


def test_overlapping_boxes_of_different_classes_both_survive():
    boxes = np.array([[10, 10, 60, 60], [12, 12, 62, 62]], dtype=np.float32)
    kept_boxes, _, _ = non_max_suppression(
        boxes, np.array([0.9, 0.8], dtype=np.float32), np.array([0, 1]),
        iou_threshold=0.45, origin_h=200, origin_w=200)
    assert len(kept_boxes) == 2


def test_suppression_clips_boxes_into_the_image():
    boxes = np.array([[-10, -10, 300, 300]], dtype=np.float32)
    kept_boxes, _, _ = non_max_suppression(
        boxes, np.array([0.9], dtype=np.float32), np.array([0]),
        iou_threshold=0.45, origin_h=100, origin_w=100)
    assert list(kept_boxes[0]) == [0, 0, 99, 99]


# ---------------------------------------------------------------- post_process

def test_post_process_drops_predictions_below_the_confidence_threshold():
    boxes = np.array([[10, 10, 60, 60], [100, 100, 150, 150]], dtype=np.float32)
    result = post_process(boxes, np.array([0.9, 0.1], dtype=np.float32), np.array([0, 0]),
                          conf_threshold=0.5, iou_threshold=0.45, origin_h=200, origin_w=200)
    assert len(result) == 1
    prediction = result[0]
    assert (prediction.x, prediction.y, prediction.width, prediction.height) == (10, 10, 50, 50)
    assert prediction.category_index == 0
    # the model's own float32 score, no longer rounded to two decimals on the way out
    assert prediction.confidence == pytest.approx(0.9)


def test_post_process_on_an_empty_prediction_returns_nothing():
    empty_boxes = np.zeros((0, 4), dtype=np.float32)
    assert post_process(empty_boxes, np.zeros(0, dtype=np.float32), np.zeros(0, dtype=int),
                        conf_threshold=0.5, iou_threshold=0.45, origin_h=10, origin_w=10) == []


def test_already_suppressed_output_keeps_the_model_s_own_precision():
    assert predictions_from_xyxy(labels=[1.0], boxes=[[10.4, 10.6, 60.4, 60.6]], scores=[0.55]) == \
        [Prediction(x=10.4, y=10.6, width=50.0, height=50.0, category_index=1, confidence=0.55)]


def test_converting_already_suppressed_output_requires_matching_lengths():
    with pytest.raises(ValueError):
        predictions_from_xyxy(labels=[1.0, 2.0], boxes=[[0.0, 0.0, 1.0, 1.0]], scores=[0.5])


# ---------------------------------------------------------------- building the containers

def test_a_box_category_becomes_a_box_detection():
    metadata = to_image_metadata([Prediction(x=10, y=20, width=30, height=40, category_index=0, confidence=0.9)], model_information(), 200, 200)
    assert len(metadata.point_detections) == 0
    detection = metadata.box_detections[0]
    assert (detection.x, detection.y, detection.width, detection.height) == (10, 20, 30, 40)
    assert (detection.category_name, detection.category_id) == ('car', 'uuid-box')
    assert detection.model_name == '1.2'
    assert detection.confidence == pytest.approx(0.9)


def test_a_point_category_becomes_the_centre_of_the_box():
    metadata = to_image_metadata([Prediction(x=100, y=100, width=40, height=40, category_index=1, confidence=0.7)], model_information(), 200, 200)
    assert len(metadata.box_detections) == 0
    detection = metadata.point_detections[0]
    assert (detection.x, detection.y) == (120, 120)
    assert detection.category_id == 'uuid-point'


def test_detections_are_clipped_to_the_image():
    metadata = to_image_metadata([Prediction(x=-20, y=-20, width=60, height=60, category_index=0, confidence=0.5)], model_information(), 200, 200)
    detection = metadata.box_detections[0]
    assert (detection.x, detection.y, detection.width, detection.height) == (0, 0, 40, 40)


@pytest.mark.parametrize('width,height', [(2, 30), (30, 2), (1, 1)])
def test_boxes_too_small_to_be_useful_are_dropped(width: int, height: int):
    metadata = to_image_metadata([Prediction(x=5, y=5, width=width, height=height, category_index=0, confidence=0.5)], model_information(), 200, 200)
    assert len(metadata) == 0


def test_a_category_type_the_node_cannot_report_is_skipped():
    classification = Category(id='uuid-cls', name='ripe', type=CategoryType.Classification)
    metadata = to_image_metadata([Prediction(x=10, y=10, width=30, height=30, category_index=0, confidence=0.5)],
                                 model_information(classification), 200, 200)
    assert len(metadata) == 0


def test_the_trainer_container_carries_the_image_id():
    result = to_detections([Prediction(x=10, y=20, width=30, height=40, category_index=0, confidence=0.9)], model_information(), 200, 200,
                           image_id='image-uuid')
    assert result.image_id == 'image-uuid'
    assert len(result.box_detections) == 1


def test_trainer_and_detector_paths_agree_on_the_same_detections():
    """The whole point of sharing this code: auto-detections and live detections must match."""
    predictions = [Prediction(x=-5, y=-5, width=60, height=60, category_index=0, confidence=0.9), Prediction(x=100, y=100, width=40, height=40, category_index=1, confidence=0.7),
                  Prediction(x=5, y=5, width=1, height=1, category_index=0, confidence=0.5)]
    metadata = to_image_metadata(predictions, model_information(), 200, 200)
    result = to_detections(predictions, model_information(), 200, 200, image_id='image-uuid')

    assert [(d.x, d.y, d.width, d.height, d.category_id) for d in result.box_detections] == \
        [(d.x, d.y, d.width, d.height, d.category_id) for d in metadata.box_detections]
    assert [(d.x, d.y, d.category_id) for d in result.point_detections] == \
        [(d.x, d.y, d.category_id) for d in metadata.point_detections]
