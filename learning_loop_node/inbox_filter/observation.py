from typing import Union
from ..detector.box_detection import BoxDetection
from ..detector.point_detection import PointDetection
from datetime import datetime, timedelta


class Observation():

    def __init__(self, detection: Union[BoxDetection, PointDetection]):
        self.detection = detection
        self.last_seen = datetime.now()

    def update_last_seen(self):
        self.last_seen = datetime.now()

    def is_older_than(self, forget_time_in_seconds):
        return self.last_seen < datetime.now() - timedelta(seconds=forget_time_in_seconds)
