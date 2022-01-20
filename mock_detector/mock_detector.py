from learning_loop_node.detector.detector import Detector
from learning_loop_node.model_information import ModelInformation
from learning_loop_node.detector.detections import Detections
from typing import Any


class MockDetector(Detector):
    def __init__(self) -> None:
        super().__init__('mocked')

    def init(self,  model_info: ModelInformation, model_root_path: str):
        self.model_info = model_info

    def evaluate(self, image: Any) -> Detections:
        return Detections()
