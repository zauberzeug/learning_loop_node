from learning_loop_node.detector import BoxDetection, Detections
from learning_loop_node import Detector, ModelInformation


class TestingDetector(Detector):
    def __init__(self) -> None:
        super().__init__('mocked')

    def init(self,  model_info: ModelInformation, model_root_path: str):
        pass

    def evaluate(self, image: bytes) -> Detections:

        return Detections(box_detections=[BoxDetection('some_category_name', 1, 2, 3, 4, 'some_model', 42)])
