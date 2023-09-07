from typing import List, Union

from learning_loop_node.data_classes import (BoxDetection, Detections,
                                             Observation, PointDetection,
                                             SegmentationDetection)


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
        return self.add_detections(Detections(segmentation_detections=seg_detections))

    def add_detections(self, detections: Detections) -> List[str]:
        causes = set()
        for detection in detections.box_detections + detections.point_detections + detections.segmentation_detections:  # P? warum nicht für classification?
            if isinstance(detection, SegmentationDetection):
                self.recent_observations.append(Observation(detection))
                causes.add('segmentation_detection')
                continue
            if not (isinstance(detection, (BoxDetection, PointDetection))):  # P? passt die änderung?
                continue

            similar = self.find_similar_observations(detection)
            if (any(similar)):
                for s in similar:
                    s.update_last_seen()
                continue
            self.recent_observations.append(Observation(detection))
            if 0.3 <= detection.confidence <= 0.6:  # TODO: P? warum nicht für segmentation/classification?
                causes.add('uncertain')
        return list(causes)

    def find_similar_observations(self, new_detection: Union[BoxDetection, PointDetection]) -> List[Observation]:

        if isinstance(new_detection, BoxDetection):
            return self.find_similar_box_observations(new_detection)

        if isinstance(new_detection, PointDetection):
            return self.find_similar_point_observations(new_detection)

        return []

    def find_similar_box_observations(self, new_detection: BoxDetection) -> List[Observation]:
        similar = []
        for observation in self.recent_observations:
            if (isinstance(observation.detection, BoxDetection) and
                observation.detection.category_name == new_detection.category_name and
                    new_detection.intersection_over_union(observation.detection) >= self.iou_threshold):
                similar.append(observation)

        return similar

    def find_similar_point_observations(self, new_detection: PointDetection) -> List[Observation]:
        similar = []
        for observation in self.recent_observations:
            if (isinstance(observation.detection, PointDetection) and
                observation.detection.category_name == new_detection.category_name and
                    new_detection.distance(observation.detection) < 10):
                similar.append(observation)

        return similar
