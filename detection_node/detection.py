class Detection:
    def __init__(self, category, x, y, width, height, net, confidence):
        self.category_name = category
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.model_name = net
        self.confidence = confidence

    @staticmethod
    def from_dict(detection: dict):
        return Detection(detection['category_name'], detection['x'], detection['y'], detection['width'], detection['height'], detection['model_name'], detection['confidence'])
