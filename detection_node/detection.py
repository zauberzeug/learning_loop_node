class Detection:
    def __init__(self, category, x, y, width, height, net, confidence):
        self.category = category
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.net = net
        self.confidence = confidence

    def _get_area(self) -> int:

        return self.width * self.height
