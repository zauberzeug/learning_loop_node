import logging
from abc import abstractmethod
from typing import Any, Optional

from ..data_classes import Detections, ModelInformation
from ..globals import GLOBALS


class DetectorLogic():

    def __init__(self, model_format: str) -> None:
        self.model_format: str = model_format
        self._model_info: Optional[ModelInformation] = None
        self.target_model: Optional[str] = None

    @property
    def model_info(self) -> ModelInformation:
        if not self._model_info:
            raise Exception('Model not loaded')
        return self._model_info

    @property
    def is_initialized(self) -> bool:
        return self._model_info is not None

    def load_model(self):
        model_info = ModelInformation.load_from_disk(f'{GLOBALS.data_folder}/model')
        if model_info is None:
            logging.warning('No model found')
            self._model_info = None
            return
        try:
            self._model_info = model_info
            self.init()
            logging.info(f'Successfully loaded model {self._model_info}')
        except Exception:
            logging.error(f'Could not init model {model_info}')
            raise

    @abstractmethod
    def init(self):
        """Initialize the model. Note that `model_info` is available as `self.model_info`"""

    @abstractmethod
    def evaluate(self, image: Any) -> Detections:
        """Evaluate the image and return the detections"""
