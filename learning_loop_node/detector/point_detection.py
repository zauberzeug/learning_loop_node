from dataclasses import dataclass
import numpy as np


@dataclass
class PointDetection:
    category_name: str
    x: int
    y: int
    model_name: str
    confidence: float

    def __init__(self, category, x, y, net, confidence):
        self.category_name = category
        self.x = x
        self.y = y
        self.model_name = net
        self.confidence = confidence

    @staticmethod
    def from_dict(detection: dict):
        return PointDetection(detection['category_name'], detection['x'], detection['y'], detection['model_name'], detection['confidence'])

    def distance(self, other: 'PointDetection') -> float:
        return np.sqrt((other.x - self.x)**2 + (other.y - self.y)**2)

    def __str__(self):
        return f'x:{int(self.x)} y: {int(self.y)}, c: {self.confidence:.2f} -> {self.category_name}'
