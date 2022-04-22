from dataclasses import dataclass


@dataclass
class BoxDetection:
    category_name: str
    x: int
    y: int
    width: int
    height: int
    model_name: str
    confidence: float
    category_id: str = ''

    def __init__(self, category_name, x, y, width, height, net, confidence, category_id=''):
        self.category_id = category_id
        self.category_name = category_name
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.model_name = net
        self.confidence = confidence

    def intersection_over_union(self, other_detection: 'BoxDetection') -> int:
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

    @staticmethod
    def from_dict(detection: dict):
        category_id = detection['category_id'] if 'category_id' in detection else ''
        return BoxDetection(detection['category_name'], detection['x'], detection['y'], detection['width'], detection['height'], detection['model_name'], detection['confidence'], category_id)

    def __str__(self):
        return f'x:{int(self.x)} y: {int(self.y)}, w: {int(self.width)} h: {int(self.height)} c: {self.confidence:.2f} -> {self.category_name}'
