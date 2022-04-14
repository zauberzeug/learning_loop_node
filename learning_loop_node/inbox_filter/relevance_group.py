from typing import List, Union
from .observation import Observation
from ..detector.box_detection import BoxDetection
from ..detector.point_detection import PointDetection
from ..detector.detections import Detections


class RelevanceGroup:
    def __init__(self):
        self.reset_time = 3600
        self.low_conf_observations: List[Observation] = []
        self.iou_threshold = 0.5

    def forget_old_detections(self):
        self.low_conf_observations = [detection
                                      for detection in self.low_conf_observations
                                      if not detection.is_older_than(self.reset_time)]

    def add_box_detections(self, box_detections: List[BoxDetection], criteria: str = '') -> List[str]:
        return self.add_detections(Detections(box_detections=box_detections), criteria=criteria)

    def add_point_detections(self, point_detections: List[PointDetection], criteria: str = '') -> List[str]:
        return self.add_detections(Detections(point_detections=point_detections), criteria=criteria)

    def add_detections(self, detections: Detections, criteria: str = '') -> List[str]:
        active_learning_causes = set()
        for detection in detections.box_detections + detections.point_detections:
            if 'novel' in criteria:
                similar = self.find_similar_observations(detection)
                if(any(similar)):
                    [s.update_last_seen() for s in similar]
                    continue
                else:
                    self.low_conf_observations.append(Observation(detection))
            if 'uncertain' in criteria and 0.3 <= detection.confidence <= .6:
                active_learning_causes.add('uncertain')
        return list(active_learning_causes)

    def find_similar_observations(self, new_detection: BoxDetection):
        return [
            observation
            for observation in self.low_conf_observations
            if observation.detection.category_name == new_detection.category_name
            and self.similar(observation.detection, new_detection)
        ]

    def similar(self, a, b) -> bool:
        if type(a) is BoxDetection and type(b) is BoxDetection:
            return a.intersection_over_union(b) >= self.iou_threshold
        if type(a) is PointDetection and type(b) is PointDetection:
            return a.distance(b) < 10
        return False
