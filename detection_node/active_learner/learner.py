from typing import List
from active_learner import detection
from icecream import ic


class Learner:
    def __init__(self):
        self.reset_time = 3600
        self.low_conf_detections = []
        self.iou = 0.9

    def forget_old_detections(self):
        ic(self.low_conf_detections)
        self.low_conf_detections = [detection
                                    for detection in self.low_conf_detections
                                    if not detection._is_older_than(self.reset_time)]

    def add_detections(self, detections: List[detection.ActiveLearnerDetection]):

        active_learning_causes = set()

        for detection in detections:
            if detection.confidence < 30 or detection.confidence > 60:
                continue

            similar_detections = self._find_similar_detection_shapes(
                self.low_conf_detections, detection)

            if(any(similar_detections)):
                ic(similar_detections)
                for sd in similar_detections:
                    sd.update_last_seen()
            else:
                self.low_conf_detections.append(detection)
                active_learning_causes.add('lowConfidence')

        return list(active_learning_causes)

    def _find_similar_detection_shapes(self, current_detections: List[detection.ActiveLearnerDetection], new_detection: detection.ActiveLearnerDetection):
        return [detection
                for detection in current_detections
                if detection.category == new_detection.category
                and detection.intersection_over_union(new_detection) >= self.iou
                ]
