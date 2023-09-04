
from learning_loop_node import DetectorLogic, ModelInformation
from learning_loop_node.data_classes import (BoxDetection, Detections, Point,
                                             PointDetection,
                                             SegmentationDetection, Shape)


class TestingDetector(DetectorLogic):
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
                point_detections=[PointDetection('some_category_name_2', 10, 12, 'some_model', .42)],
                seg_detections=[SegmentationDetection(
                    'some_category_name_3', Shape(points=[Point(1, 1)]), 'some_model', .42)]
            )

        return Detections(
            box_detections=[BoxDetection('some_category_name', 1, 2, 3, 4, 'some_model', .42)],
            point_detections=[PointDetection('some_category_name_2', 10, 12, 'some_model', .42)]
        )
