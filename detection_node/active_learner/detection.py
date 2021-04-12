from detection import Detection
from datetime import datetime, timedelta


class ActiveLearnerDetection(Detection):
    def __init__(self, detection: Detection):
        category_name = detection.category_name
        x = detection.x
        y = detection.y
        width = detection.width
        height = detection.height
        model_name = detection.model_name
        confidence = detection.confidence
        super().__init__(category_name, x, y, width, height, model_name, confidence)
        self.last_seen = datetime.now()

    def update_last_seen(self):
        self.last_seen = datetime.now()

    def _is_older_than(self, forget_time_in_seconds):
        return self.last_seen < datetime.now() - timedelta(seconds=forget_time_in_seconds)
