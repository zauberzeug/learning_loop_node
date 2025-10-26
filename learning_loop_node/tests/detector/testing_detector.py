import logging
from typing import List

import numpy as np

from learning_loop_node.data_classes import ImagesMetadata

from ...data_classes import ImageMetadata
from ...detector.detector_logic import DetectorLogic
from ..test_helper import get_dummy_metadata


class TestingDetectorLogic(DetectorLogic):
    __test__ = False

    def __init__(self) -> None:
        super().__init__('mocked')
        self.det_to_return: ImageMetadata = get_dummy_metadata()

    def init(self) -> None:
        pass

    def evaluate(self, image: np.ndarray) -> ImageMetadata:
        logging.info('evaluating')
        return self.det_to_return

    def batch_evaluate(self, images: List[np.ndarray]) -> ImagesMetadata:
        raise NotImplementedError()
