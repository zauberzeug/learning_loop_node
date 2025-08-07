import logging
from abc import abstractmethod
from typing import List, Optional

from ..data_classes import ImageMetadata, ImagesMetadata, ModelInformation
from ..globals import GLOBALS
from .exceptions import NodeNeedsRestartError


class DetectorLogic():

    def __init__(self, model_format: str) -> None:
        self.model_format: str = model_format
        self.model_info: Optional[ModelInformation] = None

        self._remaining_init_attempts: int = 2

    async def soft_reload(self):
        self.model_info = None

    def load_model_info_and_init_model(self):
        logging.info('Loading model from %s', GLOBALS.data_folder)
        self.model_info = ModelInformation.load_from_disk(f'{GLOBALS.data_folder}/model')
        if self.model_info is None:
            logging.error('No model found')
            self.model_info = None
            return

        try:
            self.init()
            logging.info('Successfully loaded model %s', self.model_info)
            self._remaining_init_attempts = 2
        except Exception:
            self._remaining_init_attempts -= 1
            self.model_info = None
            logging.error('Could not init model %s. Retries left: %s', self.model_info, self._remaining_init_attempts)
            if self._remaining_init_attempts == 0:
                raise NodeNeedsRestartError('Could not init model') from None
            raise

    @abstractmethod
    def init(self):
        """Called when a (new) model was loaded. Initialize the model. Model information available via `self.model_info`"""

    def evaluate_with_all_info(self, image: bytes, tags: List[str], source: Optional[str] = None, creation_date: Optional[str] = None) -> ImageMetadata:  # pylint: disable=unused-argument
        """Called by the detector node when an image should be evaluated (REST or SocketIO).
        Tags, source come from the caller and may be used in this function. 
        By default, this function simply calls `evaluate`"""
        return self.evaluate(image)

    @abstractmethod
    def evaluate(self, image: bytes) -> ImageMetadata:
        """Evaluate the image and return the detections.
        The object should return empty detections if it is not initialized"""

    @abstractmethod
    def batch_evaluate(self, images: List[bytes]) -> ImagesMetadata:
        """Evaluate a batch of images and return the detections.
        The object should return empty detections if it is not initialized"""
