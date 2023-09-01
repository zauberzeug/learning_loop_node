from datetime import datetime, timedelta
from typing import Union

from learning_loop_node.data_classes.detections import (BoxDetection,
                                                        PointDetection,
                                                        SegmentationDetection)


class Observation():

    def __init__(self, detection: Union[BoxDetection, PointDetection, SegmentationDetection]):
        self.detection = detection
        self.last_seen = datetime.now()

    def update_last_seen(self):
        self.last_seen = datetime.now()

    def is_older_than(self, forget_time_in_seconds):
        return self.last_seen < datetime.now() - timedelta(seconds=forget_time_in_seconds)
