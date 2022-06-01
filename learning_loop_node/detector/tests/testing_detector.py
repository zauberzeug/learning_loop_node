import logging
from learning_loop_node.detector import BoxDetection, PointDetection, SegmentationDetection, Shape, Point, Detections
from learning_loop_node import Detector, ModelInformation


class TestingDetector(Detector):
    __test__ = False

    def __init__(self, segmentation_detections: bool = False) -> None:
        super().__init__('mocked')
        self.segmentation_detections = segmentation_detections

    def init(self,  model_info: ModelInformation,  model_root_path: str):
        self.model_info = model_info

    def evaluate(self, image: bytes) -> Detections:
        if self.segmentation_detections:
            return Detections(
                box_detections=[BoxDetection('some_category_name', 1, 2, 3, 4, 'some_model', .42)],
                point_detections=[PointDetection('some_category_name', 10, 12, 'some_model', .42)],
                segmentation_detections=[SegmentationDetection(
                    'some_category_name', Shape(points=[Point(1, 1)]), 'some_model', .42)]
            )

        return Detections(
            box_detections=[BoxDetection('some_category_name', 1, 2, 3, 4, 'some_model', .42)],
            point_detections=[PointDetection('some_category_name', 10, 12, 'some_model', .42)]
        )
