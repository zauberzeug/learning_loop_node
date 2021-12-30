from typing import Optional, Union, Any
import os
import json
import logging
from learning_loop_node.model_information import ModelInformation
from learning_loop_node.globals import GLOBALS
from learning_loop_node.detector.detections import Detections


class Detector():

    def __init__(self, model_format: str) -> None:
        self.model_format: str = model_format
        self.current_model: Optional[ModelInformation] = None
        self.target_model: Optional[str] = None

    def load_model(self):
        try:
            model_root_path = f'{GLOBALS.data_folder}/model'
            model_info_file_path = f'{model_root_path}/model.json'
            if not os.path.exists(model_info_file_path):
                raise FileExistsError(f"File '{model_info_file_path}' does not exist.")
            with open(model_info_file_path, 'r') as f:
                try:
                    content = json.load(f)
                except:
                    raise Exception(f"could not read model information from file '{model_info_file_path}'")
                try:
                    model_information = ModelInformation.parse_obj(content)
                except Exception as e:
                    raise Exception(
                        f"could not parse model information from file '{model_info_file_path}'. \n {str(e)}")
            try:
                self.init(model_information, model_root_path)
                self.current_model = model_information
                logging.info(f'Successfully loaded model {self.current_model}')
            except Exception:
                logging.error(f'Could not init model {model_information}')
                raise

        except Exception:
            self.current_model = None
            logging.exception('An error occured during loading model.')

    def init(self,  model_info: ModelInformation, model_root_path: str):
        raise NotImplementedError()

    def evaluate(self, image: Any) -> Detections:
        raise NotImplementedError()
