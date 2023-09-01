from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Union

import numpy as np


@dataclass
class BoxDetection:
    category_name: str
    x: int
    y: int
    width: int
    height: int
    model_name: str
    confidence: float
    category_id: str = ''

    def __init__(self, category_name, x, y, width, height, net, confidence, category_id=''):
        self.category_id = category_id
        self.category_name = category_name
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.model_name = net
        self.confidence = confidence

    def intersection_over_union(self, other_detection: 'BoxDetection') -> float:
        # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        xA = max(self.x, other_detection.x)
        yA = max(self.y, other_detection.y)
        xB = min(self.x + self.width, other_detection.x + other_detection.width)
        yB = min(self.y + self.height, other_detection.y +
                 other_detection.height)

        interArea = max(xB - xA, 0) * max(yB - yA, 0)
        union = float(self._get_area() + other_detection._get_area() - interArea)
        if union == 0:
            print("WARNING: something went wrong while calculating iou")
            return 0

        return interArea / union

    def _get_area(self) -> int:
        return self.width * self.height

    @staticmethod
    def from_dict(detection: dict):
        category_id = detection['category_id'] if 'category_id' in detection else ''
        return BoxDetection(
            detection['category_name'],
            detection['x'],
            detection['y'],
            detection['width'],
            detection['height'],
            detection['model_name'],
            detection['confidence'],
            category_id)

    def __str__(self):
        return f'x:{int(self.x)} y: {int(self.y)}, w: {int(self.width)} h: {int(self.height)} c: {self.confidence:.2f} -> {self.category_name}'


@dataclass
class PointDetection:
    category_name: str
    x: int
    y: int
    model_name: str
    confidence: float
    category_id: str = ''

    def __init__(self, category_name, x, y, net, confidence, category_id=''):
        self.category_name = category_name
        self.x = x
        self.y = y
        self.model_name = net
        self.confidence = confidence
        self.category_id = category_id

    @staticmethod
    def from_dict(detection: dict):
        category_id = detection['category_id'] if 'category_id' in detection else ''
        return PointDetection(
            detection['category_name'],
            detection['x'],
            detection['y'],
            detection['model_name'],
            detection['confidence'],
            category_id)

    def distance(self, other: 'PointDetection') -> float:
        return np.sqrt((other.x - self.x)**2 + (other.y - self.y)**2)

    def __str__(self):
        return f'x:{int(self.x)} y: {int(self.y)}, c: {self.confidence:.2f} -> {self.category_name}'


@dataclass
class ClassificationDetection:
    category_name: str
    model_name: str
    confidence: float
    category_id: str = ''

    @staticmethod
    def from_dict(detection: dict):
        category_id = detection['category_id'] if 'category_id' in detection else ''
        return ClassificationDetection(
            detection['category_name'],
            detection['model_name'],
            detection['confidence'],
            category_id)

    def __str__(self):
        return f'c: {self.confidence:.2f} -> {self.category_name}'


@dataclass
class Point:
    x: int
    y: int

    def __str__(self):
        return f'x:{int(self.x)} y: {int(self.y)}'


@dataclass
class Shape:
    points: List[Point]

    def __str__(self):
        return ', '. join([str(s) for s in self.points])


@dataclass
class SegmentationDetection:
    category_name: str
    shape: Union[Shape, str]
    model_name: str
    confidence: float
    category_id: str = ''

    @staticmethod
    def from_dict(detection: dict):
        category_id = detection['category_id'] if 'category_id' in detection else ''
        return SegmentationDetection(
            detection['category_name'],
            detection['shape'],
            detection['model_name'],
            detection['confidence'],
            category_id)

    def __str__(self):
        return f'shape:{str(self.shape)}, c: {self.confidence:.2f} -> {self.category_name}'


@dataclass
class Detections:
    box_detections: List[BoxDetection] = field(default_factory=list)
    point_detections: List[PointDetection] = field(default_factory=list)
    seg_detections: List[SegmentationDetection] = field(default_factory=list)
    classification_detections: List[ClassificationDetection] = field(default_factory=list)
    tags: Optional[List[str]] = field(default_factory=list)
    date: Optional[str] = datetime.now().isoformat(sep='_', timespec='milliseconds')

    def __len__(self):
        return len(self.box_detections) + len(self.point_detections) + len(self.seg_detections) + len(self.classification_detections)
