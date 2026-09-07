"""Model-agnostic detection postprocessing."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..data_classes import (
    BoxDetection,
    Detections,
    ImageMetadata,
    ModelInformation,
    PointDetection,
)
from ..enums import CategoryType
from .categories import category_by_index
from .geometry import clip_box, clip_point

logger = logging.getLogger(__name__)

MIN_BOX_SIZE: int = 2


@dataclass(kw_only=True, slots=True, frozen=True)
class Prediction:
    """One surviving prediction in the model's coordinates: top-left corner and size in pixels,
    unrounded, and the category as an index into :attr:`ModelInformation.categories`."""

    x: float
    y: float
    width: float
    height: float
    category_index: int
    confidence: float


def post_process(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    *,
    conf_threshold: float,
    iou_threshold: float,
    origin_h: int,
    origin_w: int,
) -> list[Prediction]:
    """Filter by confidence, run NMS, return what survives."""
    mask = scores > conf_threshold
    boxes = boxes[mask].copy()
    scores = scores[mask]
    classes = classes[mask]

    if len(scores) == 0:
        return []

    boxes, scores, classes = non_max_suppression(
        boxes, scores, classes,
        iou_threshold=iou_threshold, origin_h=origin_h, origin_w=origin_w)

    return predictions_from_xyxy(labels=classes, boxes=boxes, scores=scores)


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    *,
    iou_threshold: float,
    origin_h: int,
    origin_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip to image bounds, sort by score descending, apply per-class NMS.

    :return: ``(boxes, scores, classes)`` — filtered arrays in the same order.
    """
    boxes = boxes.copy()
    boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h - 1)

    order = np.argsort(-scores)
    boxes = boxes[order]
    scores = scores[order]
    classes = classes[order]

    keep_indices: list[int] = []
    for cls in np.unique(classes):
        cls_mask = classes == cls
        cls_indices = np.where(cls_mask)[0]
        cls_boxes = boxes[cls_mask]

        while len(cls_indices) > 0:
            keep_indices.append(cls_indices[0])
            if len(cls_indices) == 1:
                break
            ious = bbox_iou(cls_boxes[0:1], cls_boxes[1:])
            keep_mask = ious.flatten() <= iou_threshold
            cls_indices = cls_indices[1:][keep_mask]
            cls_boxes = cls_boxes[1:][keep_mask]

    keep_indices = sorted(keep_indices)
    return boxes[keep_indices], scores[keep_indices], classes[keep_indices]


def bbox_iou(
    box1: np.ndarray,
    box2: np.ndarray,
) -> np.ndarray:
    """Compute IoU between box1 (1x4) and box2 (Nx4), both in x1y1x2y2 format."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y2 = np.minimum(b1_y2, b2_y2)

    inter_area = np.clip(inter_x2 - inter_x1 + 1, 0, None) * np.clip(inter_y2 - inter_y1 + 1, 0, None)
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)


def predictions_from_xyxy(
    *,
    labels: Sequence[float],
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
) -> list[Prediction]:
    """Convert xyxy model output into predictions, for models that suppress their own overlaps.

    Corners stay unrounded: :func:`clip_box` rounds once, when the box becomes a
    :class:`BoxDetection`.
    """
    return [Prediction(x=float(x1), y=float(y1), width=float(x2 - x1), height=float(y2 - y1),
                       category_index=int(label), confidence=float(score))
            for label, (x1, y1, x2, y2), score in zip(labels, boxes, scores, strict=True)]


def to_image_metadata(
    predictions: list[Prediction],
    model_information: ModelInformation,
    im_height: int,
    im_width: int,
) -> ImageMetadata:
    """Build the container a *detector* node reports."""
    image_metadata = ImageMetadata()
    _append_predictions(image_metadata, predictions, model_information, im_height, im_width)
    return image_metadata


def to_detections(
    predictions: list[Prediction],
    model_information: ModelInformation,
    im_height: int,
    im_width: int,
    *,
    image_id: str | None = None,
) -> Detections:
    """Build the container a *trainer*'s auto-detection pass reports."""
    result = Detections(image_id=image_id)
    _append_predictions(result, predictions, model_information, im_height, im_width)
    return result


def _append_predictions(
    target: ImageMetadata | Detections,
    predictions: list[Prediction],
    model_information: ModelInformation,
    im_height: int,
    im_width: int,
) -> None:
    """Resolve each prediction's category and append it to ``target``, clipped to the image."""
    skipped_predictions = []

    for prediction in predictions:
        category = category_by_index(model_information, prediction.category_index)
        if prediction.width <= MIN_BOX_SIZE or prediction.height <= MIN_BOX_SIZE:
            skipped_predictions.append((category.name, prediction))
            continue
        if category.type == CategoryType.Box:
            clipped_x1, clipped_y1, clipped_w, clipped_h = clip_box(
                x1=prediction.x,
                y1=prediction.y,
                width=prediction.width,
                height=prediction.height,
                img_width=im_width,
                img_height=im_height,
            )
            target.box_detections.append(
                BoxDetection(
                    category_name=category.name,
                    x=clipped_x1,
                    y=clipped_y1,
                    width=clipped_w,
                    height=clipped_h,
                    category_id=category.id,
                    model_name=model_information.version,
                    confidence=prediction.confidence,
                )
            )
        elif category.type == CategoryType.Point:
            cx, cy = prediction.x + prediction.width / 2, prediction.y + prediction.height / 2
            cx, cy = clip_point(cx, cy, im_width, im_height)
            target.point_detections.append(
                PointDetection(
                    category_name=category.name,
                    x=cx,
                    y=cy,
                    category_id=category.id,
                    model_name=model_information.version,
                    confidence=prediction.confidence,
                )
            )
        else:
            logger.warning('Unsupported category type %s for category %s', category.type, category.name)

    if skipped_predictions:
        log_msg = '\n'.join([str(p) for p in skipped_predictions])
        logger.warning('Removed %d small detections from result: \n%s', len(skipped_predictions), log_msg)
