from typing import List

from learning_loop_node.data_classes import ImageMetadata, ImagesMetadata, ModelInformation
from learning_loop_node.detector.detector_logic import DetectorLogic


class MockDetector(DetectorLogic):

    def __init__(self, model_info: ModelInformation) -> None:
        pass

    def evaluate(self, image: bytes) -> ImageMetadata:
        return ImageMetadata()

    def batch_evaluate(self, images: List[bytes]) -> ImagesMetadata:
        raise NotImplementedError()


class MockDetectorFactory:
    model_format = 'mocked'

    async def build(self, model_info: ModelInformation) -> MockDetector:
        return MockDetector(model_info)
