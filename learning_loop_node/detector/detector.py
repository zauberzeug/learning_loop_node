from typing import Optional, Union, Any
import logging
from learning_loop_node.model_information import ModelInformation
from learning_loop_node.detector.detections import Detections
from learning_loop_node.globals import GLOBALS


class Detector():

    def __init__(self, model_format: str) -> None:
        self.model_format: str = model_format
        self.current_model: Optional[ModelInformation] = None
        self.target_model: Optional[str] = None

    def load_model(self):
        try:
            model_information = ModelInformation.load_from_disk(f'{GLOBALS.data_folder}/model')
            try:
                self.init(model_information)
                self.current_model = model_information
                logging.info(f'Successfully loaded model {self.current_model}')
            except Exception:
                logging.error(f'Could not init model {model_information}')
                raise
        except Exception:
            self.current_model = None
            logging.exception('An error occured during loading model.')

    def init(self,  model_info: ModelInformation):
        raise NotImplementedError()

    def evaluate(self, image: Any) -> Detections:
        raise NotImplementedError()
