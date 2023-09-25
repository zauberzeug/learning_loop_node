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

        # return Detections(
        #     box_detections=[BoxDetection(category_name='some_category_name', x=1, y=2, height=3, width=4,
        #                                  model_name='some_model', confidence=.42, category_id='some_id')],
        #     point_detections=[PointDetection(category_name='some_category_name_2', x=10, y=12,
        #                                      model_name='some_model', confidence=.42, category_id='some_id')]
        # )
