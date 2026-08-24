from ..data_classes import (
    BoxDetection,
    ClassificationDetection,
    Detections,
    Point,
    PointDetection,
    SegmentationDetection,
    Shape,
)
from ..data_classes.image_metadata import ImageMetadata


def get_dummy_detections() -> Detections:
    return Detections(
        box_detections=[
            BoxDetection(category_name='some_category_name', x=1, y=2, height=3, width=4,
                         model_name='some_model', confidence=.42, category_id='some_id')],
        point_detections=[
            PointDetection(category_name='some_category_name_2', x=10, y=12,
                           model_name='some_model', confidence=.42, category_id='some_id_2')],
        segmentation_detections=[
            SegmentationDetection(category_name='some_category_name_3',
                                  shape=Shape(points=[Point(x=1, y=1)]),
                                  model_name='some_model', confidence=.42,
                                  category_id='some_id_3')],
        classification_detections=[
            ClassificationDetection(category_name='some_category_name_4', model_name='some_model',
                                    confidence=.42, category_id='some_id_4')])


def get_dummy_metadata() -> ImageMetadata:
    detections = get_dummy_detections()
    return ImageMetadata(box_detections=detections.box_detections,
                         point_detections=detections.point_detections,
                         segmentation_detections=detections.segmentation_detections,
                         classification_detections=detections.classification_detections)
