import logging

import numpy as np

from ...data_classes import ImageMetadata
from ...detector.detector_logic import DetectorLogic
from ..test_helper import get_dummy_detections


class TestingDetectorLogic(DetectorLogic):
    __test__ = False

    def __init__(self) -> None:
        super().__init__('mocked')
        self.det_to_return: ImageMetadata = get_dummy_detections()

    def init(self) -> None:
        pass

    def evaluate(self, image: np.ndarray) -> ImageMetadata:
        logging.info('evaluating')
        return self.det_to_return
