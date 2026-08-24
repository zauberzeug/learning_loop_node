import logging
from typing import List

import numpy as np

from ..data_classes import ImageMetadata, ImagesMetadata, ModelInformation
from ..detector.detector_logic import DetectorLogic
from .detections import get_dummy_metadata


class TestingDetectorLogic(DetectorLogic):
    __test__ = False

    def __init__(self, model_info: ModelInformation) -> None:
        self.det_to_return: ImageMetadata = get_dummy_metadata()

    def evaluate(self, image: np.ndarray) -> ImageMetadata:
        logging.info('evaluating')
        return self.det_to_return

    def batch_evaluate(self, images: List[np.ndarray]) -> ImagesMetadata:
        raise NotImplementedError()


class TestingDetectorFactory:
    __test__ = False
    model_format = 'mocked'

    async def build(self, model_info: ModelInformation) -> TestingDetectorLogic:
        return TestingDetectorLogic(model_info)
