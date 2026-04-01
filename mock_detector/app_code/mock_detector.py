from typing import final

import numpy as np

from learning_loop_node.data_classes import (
    ImageMetadata,
    ImagesMetadata,
    ModelInformation,
)
from learning_loop_node.detector.detector_logic import DetectorLogic, DetectorLogicFactory


@final
class MockDetector(DetectorLogic):

    def __init__(self, model_info: ModelInformation) -> None:
        pass

    def evaluate(self, image: np.ndarray) -> ImageMetadata:
        return ImageMetadata()

    def batch_evaluate(self, images: list[np.ndarray]) -> ImagesMetadata:
        raise NotImplementedError()


@final
class MockDetectorFactory(DetectorLogicFactory):

    @property
    def model_format(self) -> str:
        return 'mocked'

    async def build(self, model_info: ModelInformation) -> MockDetector:
        return MockDetector(model_info)
