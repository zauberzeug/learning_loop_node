from typing import List
from dataclasses import dataclass, field

from learning_loop_node.detector.box_detection import BoxDetection
from learning_loop_node.detector.point_detection import PointDetection


@dataclass
class Detections:
    box_detections: List[BoxDetection] = field(default_factory=list)
    point_detections: List[PointDetection] = field(default_factory=list)
