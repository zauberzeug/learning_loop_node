from detection import Detection
from datetime import datetime, timedelta


class ActiveLearnerDetection(Detection):
    def __init__(self, category, x, y, width, height, net, confidence):
        super().__init__(category, x, y, width, height, net, confidence)
        self.last_seen = datetime.now()

    def update_last_seen(self):
        self.last_seen = datetime.now()

    def _is_older_than(self, forget_time_in_seconds):
        return self.last_seen < datetime.now() - timedelta(seconds=forget_time_in_seconds)

    def intersection_over_union(self, other_detection: 'Detection') -> int:

        # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        xA = max(self.x, other_detection.x)
        yA = max(self.y, other_detection.y)
        xB = min(self.x + self.width, other_detection.x + other_detection.width)
        yB = min(self.y + self.height, other_detection.y +
                 other_detection.height)

        interArea = max(xB - xA, 0) * max(yB - yA, 0)
        union = float(self._get_area() + other_detection._get_area() - interArea)
        if union == 0:
            print("WARNING: something went wrong while calculating iou")
            return 0

        return interArea / union

    def _get_area(self) -> int:

        return self.width * self.height
