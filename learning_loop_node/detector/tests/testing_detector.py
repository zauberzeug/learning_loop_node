import logging

import numpy as np

from learning_loop_node import DetectorLogic
from learning_loop_node.conftest import get_dummy_detections
from learning_loop_node.data_classes import Detections


class TestingDetectorLogic(DetectorLogic):
    __test__ = False

    def __init__(self) -> None:
        super().__init__('mocked')
        self.det_to_return: Detections = get_dummy_detections()

    def init(self) -> None:
        pass

    def evaluate(self, image: np.ndarray) -> Detections:
        logging.info('evaluating')
        return self.det_to_return
