from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass, field

from learning_loop_node.detector.box_detection import BoxDetection
from learning_loop_node.detector.point_detection import PointDetection
from learning_loop_node.detector.segmentation_detection import SegmentationDetection


@dataclass
class Detections:
    box_detections: List[BoxDetection] = field(default_factory=list)
    point_detections: List[PointDetection] = field(default_factory=list)
    segmentation_detections: List[SegmentationDetection] = field(default_factory=list)
    tags: Optional[List[str]] = field(default_factory=list)
    date: Optional[str] = datetime.now().isoformat(sep='_', timespec='milliseconds')

    def __len__(self):
        return len(self.box_detections) + len(self.point_detections) + len(self.segmentation_detections)
