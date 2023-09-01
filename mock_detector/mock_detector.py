from typing import Any

from learning_loop_node.data_classes import Detections
from learning_loop_node.detector.detector import Detector
from learning_loop_node.model_information import ModelInformation


class MockDetector(Detector):
    def __init__(self) -> None:
        super().__init__('mocked')

    def init(self,  model_info: ModelInformation):
        self.model_info = model_info

    def evaluate(self, image: Any) -> Detections:
        return Detections()
