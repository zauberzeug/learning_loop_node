from dataclasses import dataclass


@dataclass
class ClassificationDetection:
    category_name: str
    model_name: str
    confidence: float
    category_id: str = ''

    @staticmethod
    def from_dict(detection: dict):
        category_id = detection['category_id'] if 'category_id' in detection else ''
        return ClassificationDetection(detection['category_name'], detection['model_name'], detection['confidence'], category_id)

    def __str__(self):
        return f'c: {self.confidence:.2f} -> {self.category_name}'
