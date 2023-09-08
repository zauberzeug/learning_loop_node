import logging

from learning_loop_node import DetectorLogic
from learning_loop_node.data_classes import Detections
from learning_loop_node.data_classes.detections import get_dummy_detections


class TestingDetectorLogic(DetectorLogic):
    __test__ = False

    def __init__(self) -> None:
        super().__init__('mocked')
        self.det_to_return = get_dummy_detections()

    def init(self):
        pass

    def evaluate(self, image: bytes) -> Detections:
        logging.info('evaluating')
        print(self.det_to_return)
        return self.det_to_return

        # return Detections(
        #     box_detections=[BoxDetection(category_name='some_category_name', x=1, y=2, height=3, width=4,
        #                                  model_name='some_model', confidence=.42, category_id='some_id')],
        #     point_detections=[PointDetection(category_name='some_category_name_2', x=10, y=12,
        #                                      model_name='some_model', confidence=.42, category_id='some_id')]
        # )
