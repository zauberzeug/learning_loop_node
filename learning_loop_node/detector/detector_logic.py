import logging
from abc import abstractmethod
from typing import List, Optional

import numpy as np

from ..data_classes import ImageMetadata, ModelInformation
from ..globals import GLOBALS
from .exceptions import NodeNeedsRestartError


class DetectorLogic():

    def __init__(self, model_format: str) -> None:
        self.model_format: str = model_format
        self._model_info: Optional[ModelInformation] = None

        self._remaining_init_attempts: int = 2

    async def soft_reload(self):
        self._model_info = None

    @property
    def model_info(self) -> ModelInformation:
        if self._model_info is None:
            raise Exception('Model not loaded')
        return self._model_info

    @property
    def is_initialized(self) -> bool:
        return self._model_info is not None

    def load_model_info_and_init_model(self):
        logging.info('Loading model from %s', GLOBALS.data_folder)
        self._model_info = ModelInformation.load_from_disk(f'{GLOBALS.data_folder}/model')
        if self._model_info is None:
            logging.error('No model found')
            self._model_info = None
        try:
            self.init()
            logging.info('Successfully loaded model %s', self._model_info)
            self._remaining_init_attempts = 2
        except Exception:
            self._remaining_init_attempts -= 1
            self._model_info = None
            logging.error('Could not init model %s. Retries left: %s', self._model_info, self._remaining_init_attempts)
            if self._remaining_init_attempts == 0:
                raise NodeNeedsRestartError('Could not init model') from None
            raise

    @abstractmethod
    def init(self):
        """Called when a (new) model was loaded. Initialize the model. Model information available via `self.model_info`"""

    def evaluate_with_all_info(self, image: np.ndarray, tags: List[str], source: Optional[str] = None, creation_date: Optional[str] = None) -> ImageMetadata:  # pylint: disable=unused-argument
        """Called by the detector node when an image should be evaluated (REST or SocketIO).
        Tags, source come from the caller and may be used in this function. 
        By default, this function simply calls `evaluate`"""
        return self.evaluate(image)

    @abstractmethod
    def evaluate(self, image: np.ndarray) -> ImageMetadata:
        """Evaluate the image and return the detections.
        The object should return empty detections if it is not initialized"""
