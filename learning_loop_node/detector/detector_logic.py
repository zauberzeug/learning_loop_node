import logging
from abc import abstractmethod
from typing import List, Optional

import numpy as np

from ..data_classes import Detections, ModelInformation
from ..globals import GLOBALS


class DetectorLogic():

    def __init__(self, model_format: str) -> None:
        self.model_format: str = model_format
        self._model_info: Optional[ModelInformation] = None

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

    def load_model(self):
        logging.info('Loading model from %s', GLOBALS.data_folder)
        model_info = ModelInformation.load_from_disk(f'{GLOBALS.data_folder}/model')
        if model_info is None:
            logging.warning('No model found')
            self._model_info = None
            return
        try:
            self._model_info = model_info
            self.init()
            logging.info('Successfully loaded model %s', self._model_info)
        except Exception:
            logging.error('Could not init model %s', model_info)
            raise

    @abstractmethod
    def init(self):
        """Called when a (new) model was loaded. Initialize the model. Model information available via `self.model_info`"""

    def evaluate_with_all_info(self, image: np.ndarray, tags: List[str], source: Optional[str] = None) -> Detections:  # pylint: disable=unused-argument
        """Called by the detector node when an image should be evaluated (REST or SocketIO).
        Tags, source come from the caller and may be used in this function. 
        By default, this function simply calls `evaluate`"""
        return self.evaluate(image)

    @abstractmethod
    def evaluate(self, image: np.ndarray) -> Detections:
        """Evaluate the image and return the detections.
        The object should return empty detections if it is not initialized"""
