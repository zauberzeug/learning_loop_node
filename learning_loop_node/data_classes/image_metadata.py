
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from .annotations import BoxAnnotation, ClassificationAnnotation, PointAnnotation
from .detections import (
    BoxDetection,
    ClassificationDetection,
    PointDetection,
    SegmentationDetection,
)

# pylint: disable=too-many-instance-attributes

KWONLY_SLOTS = {'kw_only': True, 'slots': True} if sys.version_info >= (3, 10) else {}


def current_datetime():
    return datetime.now().isoformat(sep='_', timespec='milliseconds')


@dataclass(**KWONLY_SLOTS)
class ImageMetadata():
    box_detections: List[BoxDetection] = field(default_factory=list, metadata={
        'description': 'List of box detections'})
    point_detections: List[PointDetection] = field(default_factory=list, metadata={
        'description': 'List of point detections'})
    segmentation_detections: List[SegmentationDetection] = field(default_factory=list, metadata={
        'description': 'List of segmentation detections'})
    classification_detections: List[ClassificationDetection] = field(default_factory=list, metadata={
        'description': 'List of classification detections'})

    box_annotations: List[BoxAnnotation] = field(default_factory=list, metadata={
        'description': 'List of box annotations'})
    point_annotations: List[PointAnnotation] = field(default_factory=list, metadata={
        'description': 'List of point annotations'})
    classification_annotation: Optional[ClassificationAnnotation] = field(default=None, metadata={
        'description': 'Classification annotation'})

    tags: List[str] = field(default_factory=list, metadata={
        'description': 'List of tags'})

    created: Optional[str] = field(default_factory=current_datetime, metadata={
        'description': 'Creation date of the image'})
    source: Optional[str] = field(default=None, metadata={
        'description': 'Source of the image'})

    def __len__(self):
        return len(self.box_detections) + len(self.point_detections) + len(self.segmentation_detections) + len(self.classification_detections)


@dataclass(**KWONLY_SLOTS)
class ImagesMetadata():
    items: List[ImageMetadata] = field(default_factory=list, metadata={'description': 'List of image metadata'})
