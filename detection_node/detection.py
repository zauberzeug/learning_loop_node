class Detection:
    def __init__(self, category, x, y, width, height, net, confidence):
        self.category_name = category
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.model_name = net
        self.confidence = confidence

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


    @staticmethod
    def from_dict(detection: dict):
        return Detection(detection['category_name'], detection['x'], detection['y'], detection['width'], detection['height'], detection['model_name'], detection['confidence'])
