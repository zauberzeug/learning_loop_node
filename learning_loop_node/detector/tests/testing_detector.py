import logging

from learning_loop_node import DetectorLogic
from learning_loop_node.data_classes import (BoxDetection,
                                             ClassificationDetection,
                                             Detections, Point, PointDetection,
                                             SegmentationDetection, Shape)


class TestingDetector(DetectorLogic):
    __test__ = False

    def __init__(self) -> None:
        super().__init__('mocked')

    def init(self):
        pass

    def load_model(self):
        pass

    def evaluate(self, image: bytes) -> Detections:
        logging.info('evaluating')
        det = Detections.dummy()
        print(det)
        return det

        # return Detections(
        #     box_detections=[BoxDetection(category_name='some_category_name', x=1, y=2, height=3, width=4,
        #                                  model_name='some_model', confidence=.42, category_id='some_id')],
        #     point_detections=[PointDetection(category_name='some_category_name_2', x=10, y=12,
        #                                      model_name='some_model', confidence=.42, category_id='some_id')]
        # )
