
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Union

import numpy as np

# pylint: disable=too-many-instance-attributes

KWONLY_SLOTS = {'kw_only': True, 'slots': True} if sys.version_info >= (3, 10) else {}


@dataclass(**KWONLY_SLOTS)
class BoxDetection():
    category_name: str
    x: int  # TODO add definition of x,y,w,h
    y: int
    width: int
    height: int
    model_name: str
    confidence: float
    category_id: str

    def intersection_over_union(self, other_detection: 'BoxDetection') -> float:
        # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        xA = max(self.x, other_detection.x)
        yA = max(self.y, other_detection.y)
        xB = min(self.x + self.width, other_detection.x + other_detection.width)
        yB = min(self.y + self.height, other_detection.y +
                 other_detection.height)

        interArea = max(xB - xA, 0) * max(yB - yA, 0)
        union = float(self._get_area() + other_detection._get_area() - interArea)  # pylint: disable=protected-access
        if union == 0:
            print("WARNING: something went wrong while calculating iou")
            return 0

        return interArea / union

    def _get_area(self) -> int:
        return self.width * self.height

    def __str__(self):
        return f'x:{int(self.x)} y: {int(self.y)}, w: {int(self.width)} h: {int(self.height)} c: {self.confidence:.2f} -> {self.category_name}'


@dataclass(**KWONLY_SLOTS)
class PointDetection():
    category_name: str
    x: int
    y: int
    model_name: str
    confidence: float
    category_id: str

    def distance(self, other: 'PointDetection') -> float:
        return np.sqrt((other.x - self.x)**2 + (other.y - self.y)**2)

    def __str__(self):
        return f'x:{int(self.x)} y: {int(self.y)}, c: {self.confidence:.2f} -> {self.category_name}'


@dataclass(**KWONLY_SLOTS)
class ClassificationDetection():
    category_name: str
    model_name: str
    confidence: float
    category_id: str

    def __str__(self):
        return f'c: {self.confidence:.2f} -> {self.category_name}'


@dataclass(**KWONLY_SLOTS)
class Point():
    x: int
    y: int

    def __str__(self):
        return f'x:{self.x} y: {self.y}'


@dataclass(**KWONLY_SLOTS)
class Shape():
    points: List[Point]

    def __str__(self):
        return ', '. join([str(s) for s in self.points])


@dataclass(**KWONLY_SLOTS)
class SegmentationDetection():
    category_name: str
    shape: Union[Shape, str]
    model_name: str
    confidence: float
    category_id: str

    def __str__(self):
        return f'shape:{str(self.shape)}, c: {self.confidence:.2f} -> {self.category_name}'


def current_datetime():
    return datetime.now().isoformat(sep='_', timespec='milliseconds')


@dataclass(**KWONLY_SLOTS)
class Detections():
    box_detections: List[BoxDetection] = field(default_factory=list)
    point_detections: List[PointDetection] = field(default_factory=list)
    segmentation_detections: List[SegmentationDetection] = field(default_factory=list)
    classification_detections: List[ClassificationDetection] = field(default_factory=list)
    tags: Optional[List[str]] = field(default_factory=list)
    date: Optional[str] = field(default_factory=current_datetime)
    image_id: Optional[str] = None  # used for detection of trainers

    def __len__(self):
        return len(self.box_detections) + len(self.point_detections) + len(self.segmentation_detections) + len(self.classification_detections)


def get_dummy_detections():
    return Detections(
        box_detections=[
            BoxDetection(category_name='some_category_name', x=1, y=2, height=3, width=4,
                         model_name='some_model', confidence=.42, category_id='some_id')],
        point_detections=[
            PointDetection(category_name='some_category_name_2', x=10, y=12,
                           model_name='some_model', confidence=.42, category_id='some_id_2')],
        segmentation_detections=[
            SegmentationDetection(category_name='some_category_name_3',
                                  shape=Shape(points=[Point(x=1, y=1)]),
                                  model_name='some_model', confidence=.42,
                                  category_id='some_id_3')],
        classification_detections=[
            ClassificationDetection(category_name='some_category_name_4', model_name='some_model',
                                    confidence=.42, category_id='some_id_4')])


# TODO make dataclass
class Observation():

    def __init__(self, detection: Union[BoxDetection, PointDetection, SegmentationDetection, ClassificationDetection]):
        self.detection = detection
        self.last_seen = datetime.now()

    def update_last_seen(self):
        self.last_seen = datetime.now()

    def is_older_than(self, forget_time_in_seconds):
        return self.last_seen < datetime.now() - timedelta(seconds=forget_time_in_seconds)
