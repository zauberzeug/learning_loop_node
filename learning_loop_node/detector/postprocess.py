"""Model-agnostic detection postprocessing.

Every detection node ends up doing the same three things with a model's raw output: drop
low-confidence predictions, suppress overlapping boxes, and turn what survives into the
loop's detection dataclasses. None of that depends on the model, so it lives here rather
than being re-derived — and re-diverging — in each node repository.

Two containers carry the same detections in this library: a detector node reports
:class:`~learning_loop_node.data_classes.image_metadata.ImageMetadata`, while a trainer's
auto-detection pass reports :class:`~learning_loop_node.data_classes.detections.Detections`.
:func:`to_image_metadata` and :func:`to_detections` build them from the same routine, so both
paths clip and filter identically.
"""

import logging
from collections import namedtuple

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
"""Boxes this small are dropped: they carry no usable information and clutter the loop."""

Detection = namedtuple('Detection', 'x y w h category probability')
"""One surviving prediction. ``x``/``y`` are the top-left corner, ``category`` is an index
into :attr:`ModelInformation.categories`."""


def post_process(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    *,
    conf_threshold: float,
    iou_threshold: float,
    origin_h: int,
    origin_w: int,
) -> list[Detection]:
    """Filter by confidence, run NMS, return a :class:`Detection` list in x/y/w/h form."""
    mask = scores > conf_threshold
    boxes = boxes[mask].copy()
    scores = scores[mask]
    classes = classes[mask]

    if len(scores) == 0:
        return []

    boxes, scores, classes = non_max_suppression(
        boxes, scores, classes,
        iou_threshold=iou_threshold, origin_h=origin_h, origin_w=origin_w)

    result = []
    for j, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        result.append(Detection(int(x1), int(y1), int(w), int(h),
                                int(classes[j]), round(float(scores[j]), 2)))
    return result


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


def detections_from_xyxy(
    *,
    labels: list[float],
    boxes: list[list[float]],
    scores: list[float],
) -> list[Detection]:
    """Convert already-suppressed model output into :class:`Detection` values.

    For nodes whose model (or a torch/ONNX op) has done the suppression already, so only the
    coordinate conversion is left. Corners are rounded rather than truncated, which is half a
    pixel more faithful than :func:`post_process` — that one keeps truncating so its output
    stays bit-identical to what detectors reported before this module existed.
    """
    result = []
    for label, box, score in zip(labels, boxes, scores, strict=True):
        x1, y1, x2, y2 = (round(value) for value in box)
        result.append(Detection(x1, y1, x2 - x1, y2 - y1, int(label), score))
    return result


def to_image_metadata(
    detections: list[Detection],
    model_information: ModelInformation,
    im_height: int,
    im_width: int,
) -> ImageMetadata:
    """Build the container a *detector* node reports from a list of detections."""
    image_metadata = ImageMetadata()
    _append_detections(image_metadata, detections, model_information, im_height, im_width)
    return image_metadata


def to_detections(
    detections: list[Detection],
    model_information: ModelInformation,
    im_height: int,
    im_width: int,
    *,
    image_id: str | None = None,
) -> Detections:
    """Build the container a *trainer*'s auto-detection pass reports."""
    result = Detections(image_id=image_id)
    _append_detections(result, detections, model_information, im_height, im_width)
    return result


def _append_detections(
    target: ImageMetadata | Detections,
    detections: list[Detection],
    model_information: ModelInformation,
    im_height: int,
    im_width: int,
) -> None:
    """Resolve each detection's category and append it to ``target``, clipped to the image."""
    skipped_detections = []

    for detection in detections:
        x, y, w, h, category_idx, probability = detection
        category = category_by_index(model_information, category_idx)
        if w <= MIN_BOX_SIZE or h <= MIN_BOX_SIZE:
            skipped_detections.append((category.name, detection))
            continue
        if category.type == CategoryType.Box:
            clipped_x1, clipped_y1, clipped_w, clipped_h = clip_box(
                x1=x,
                y1=y,
                width=w,
                height=h,
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
                    confidence=probability,
                )
            )
        elif category.type == CategoryType.Point:
            cx, cy = x + w / 2, y + h / 2
            cx, cy = clip_point(cx, cy, im_width, im_height)
            target.point_detections.append(
                PointDetection(
                    category_name=category.name,
                    x=cx,
                    y=cy,
                    category_id=category.id,
                    model_name=model_information.version,
                    confidence=probability,
                )
            )
        else:
            logger.warning('Unsupported category type %s for category %s', category.type, category.name)

    if skipped_detections:
        log_msg = '\n'.join([str(d) for d in skipped_detections])
        logger.warning('Removed %d small detections from result: \n%s', len(skipped_detections), log_msg)
