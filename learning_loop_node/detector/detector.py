import logging
from abc import abstractmethod
from typing import Any, Optional

from learning_loop_node.data_classes import Detections, ModelInformation
from learning_loop_node.globals import GLOBALS


class Detector():

    def __init__(self, model_format: str) -> None:
        self.model_format: str = model_format
        self.model_info: Optional[ModelInformation] = None
        self.target_model: Optional[str] = None

    def load_model(self):
        try:
            model_information = ModelInformation.load_from_disk(f'{GLOBALS.data_folder}/model')
            try:
                self.init(model_information)
                self.model_info = model_information
                logging.info(f'Successfully loaded model {self.model_info}')
            except Exception:
                logging.error(f'Could not init model {model_information}')
                raise
        except Exception:
            self.model_info = None
            logging.exception('An error occured during loading model.')

    @abstractmethod
    def init(self,  model_info: ModelInformation):
        pass

    @abstractmethod
    def evaluate(self, image: Any) -> Detections:
        pass
