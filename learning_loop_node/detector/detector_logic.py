import logging
from abc import abstractmethod
from typing import List, Optional

import numpy as np

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
        """
        Load model information from disk and initialize the model.

        The detector node uses a lock to make sure that this is not called 
        concurrently with evaluate() or batch_evaluate(). 
        """
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
        """
        Initialize the model.

        Called when a (new) model was loaded. 
        Model information available via `self.model_info`
        The detector node uses a lock to make sure that this is not called
        concurrently with evaluate() or batch_evaluate(). 
        """

    @abstractmethod
    def evaluate(self, image: np.ndarray) -> ImageMetadata:
        """
        Evaluate the image and return the detections.

        Called by the detector node when an image should be evaluated (REST or SocketIO).
        The resulting detections should be stored in the ImageMetadata.
        Tags stored in the ImageMetadata will be uploaded to the learning loop.
        The function should return empty metadata if the detector is not initialized.
        """

    @abstractmethod
    def batch_evaluate(self, images: List[np.ndarray]) -> ImagesMetadata:
        """
        Evaluate a batch of images and return the detections.

        The resulting detections per image should be stored in the ImagesMetadata.
        Tags stored in the ImagesMetadata will be uploaded to the learning loop.
        The function should return empty metadata if the detector is not initialized.
        """
