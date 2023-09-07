from typing import Any

from learning_loop_node.data_classes import Detections
from learning_loop_node.detector.detector_logic import DetectorLogic


class MockDetector(DetectorLogic):
    def __init__(self, model_format) -> None:
        super().__init__(model_format=model_format)

    @property
    def is_initialized(self) -> bool:
        return True

    def init(self) -> None:
        pass

    def evaluate(self, image: Any) -> Detections:
        print('------------------ evaluate')
        return Detections()
