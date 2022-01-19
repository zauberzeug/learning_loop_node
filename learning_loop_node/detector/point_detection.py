from dataclasses import dataclass
import numpy as np


@dataclass
class PointDetection:
    category_name: str
    x: int
    y: int
    model_name: str
    confidence: float
    category_id: str = ''

    def __init__(self, category_name, x, y, net, confidence, category_id=''):
        self.category_name = category_name
        self.x = x
        self.y = y
        self.model_name = net
        self.confidence = confidence
        self.category_id = category_id

    @staticmethod
    def from_dict(detection: dict):
        category_id = detection['category_id'] if 'category_id' in detection else ''
        return PointDetection(detection['category_name'], detection['x'], detection['y'], detection['model_name'], detection['confidence'], category_id)


    def distance(self, other: 'PointDetection') -> float:
        return np.sqrt((other.x - self.x)**2 + (other.y - self.y)**2)

    def __str__(self):
        return f'x:{int(self.x)} y: {int(self.y)}, c: {self.confidence:.2f} -> {self.category_name}'
