from abc import abstractmethod
from typing import List, Protocol

import numpy as np

from ..data_classes import ImageMetadata, ImagesMetadata, ModelInformation


class DetectorLogicFactory(Protocol):
    """Protocol for building DetectorLogic instances.

    The factory controls how the detector is constructed — implementations
    can build synchronously or offload heavy work to a thread pool.
    """
    model_format: str

    async def build(self, model_info: ModelInformation) -> 'DetectorLogic': ...


class DetectorLogic():
    """Pure interface for detector implementations.

    Subclasses receive the ModelInformation via their constructor (called by the
    DetectorLogic factory in DetectorNode) and are free to store or ignore it.
    """

    @abstractmethod
    def evaluate(self, image: np.ndarray) -> ImageMetadata:
        """Evaluate the image and return the detections.

        Called by the detector node when an image should be evaluated (REST or SocketIO).
        The resulting detections should be stored in the ImageMetadata.
        Tags stored in the ImageMetadata will be uploaded to the learning loop.
        """

    @abstractmethod
    def batch_evaluate(self, images: List[np.ndarray]) -> ImagesMetadata:
        """Evaluate a batch of images and return the detections.

        The resulting detections per image should be stored in the ImagesMetadata.
        Tags stored in the ImagesMetadata will be uploaded to the learning loop.
        """
