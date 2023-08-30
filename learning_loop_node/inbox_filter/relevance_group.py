from typing import List

from learning_loop_node.detector.detections import (BoxDetection, Detections,
                                                    PointDetection,
                                                    SegmentationDetection)
from learning_loop_node.inbox_filter.observation import Observation


class RelevanceGroup:
    def __init__(self):
        self.reset_time = 3600
        self.recent_observations: List[Observation] = []
        self.iou_threshold = 0.5

    def forget_old_detections(self):
        self.recent_observations = [detection
                                    for detection in self.recent_observations
                                    if not detection.is_older_than(self.reset_time)]

    def add_box_detections(self, box_detections: List[BoxDetection]) -> List[str]:
        return self.add_detections(Detections(box_detections=box_detections))

    def add_point_detections(self, point_detections: List[PointDetection]) -> List[str]:
        return self.add_detections(Detections(point_detections=point_detections))

    def add_segmentation_detections(self, seg_detections: List[SegmentationDetection]) -> List[str]:
        return self.add_detections(Detections(seg_detections=seg_detections))

    def add_detections(self, detections: Detections) -> List[str]:
        causes = set()
        for detection in detections.box_detections + detections.point_detections + detections.seg_detections:
            if isinstance(detection, SegmentationDetection):
                self.recent_observations.append(Observation(detection))
                causes.add('segmentation_detection')
                continue
            similar = self.find_similar_observations(detection)
            if (any(similar)):
                [s.update_last_seen() for s in similar]
                continue
            else:
                self.recent_observations.append(Observation(detection))
                if 0.3 <= detection.confidence <= .6:
                    causes.add('uncertain')
        return list(causes)

    def find_similar_observations(self, new_detection: BoxDetection):
        return [
            observation
            for observation in self.recent_observations
            if observation.detection.category_name == new_detection.category_name
            and self.similar(observation.detection, new_detection)
        ]

    def similar(self, a, b) -> bool:
        if type(a) is BoxDetection and type(b) is BoxDetection:
            return a.intersection_over_union(b) >= self.iou_threshold
        if type(a) is PointDetection and type(b) is PointDetection:
            return a.distance(b) < 10
        return False
