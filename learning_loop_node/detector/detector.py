from typing import Union, Any
from learning_loop_node.model_information import ModelInformation
from learning_loop_node.globals import GLOBALS
import os
import json
from icecream import ic
from learning_loop_node.detector.detections import Detections
import logging


class Detector():
    current_model: Union[ModelInformation, None]
    target_model_id: Union[str, None] = None
    model_format: str

    def __init__(self, model_format: str) -> None:
        self.model_format = model_format
        self.current_model = None

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
                    ic(model_information)
                except Exception as e:
                    raise Exception(
                        f"could not parse model information from file '{model_info_file_path}'. \n {str(e)}")
            try:
                self.init(model_information, model_root_path)
            except Exception:
                logging.error('Could not init model {model_information}')
                raise
        except Exception:
            self.current_model = None
            logging.error('An error occured during loading model.')
            raise

        self.current_model = model_information
        logging.info(f'Successfully loaded model. Current Model id is : { self.current_model.id}')

    def init(self,  model_info: ModelInformation, model_root_path: str):
        raise NotImplementedError()

    def evaluate(self, image: Any) -> Detections:
        raise NotImplementedError()
