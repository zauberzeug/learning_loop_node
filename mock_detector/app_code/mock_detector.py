from typing import List, Optional

from learning_loop_node.data_classes import ImageMetadata, ImagesMetadata
from learning_loop_node.detector.detector_logic import DetectorLogic


class MockDetector(DetectorLogic):
    def __init__(self, model_format) -> None:
        super().__init__(model_format=model_format)

    def init(self) -> None:
        pass

    def evaluate(self,
                 image: bytes,
                 tags: List[str],
                 source: Optional[str] = None,
                 creation_date: Optional[str] = None) -> ImageMetadata:
        return ImageMetadata()

    def batch_evaluate(self,
                       images: List[bytes],
                       tags: List[str],
                       source: str | None = None,
                       creation_date: str | None = None) -> ImagesMetadata:
        raise NotImplementedError()
