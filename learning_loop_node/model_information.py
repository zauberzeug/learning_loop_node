from pydantic.main import BaseModel
from typing import List, Optional
from learning_loop_node.context import Context
from learning_loop_node.data_classes.category import Category
import os
import json


class ModelInformation(BaseModel):
    id: str
    host: Optional[str]
    organization: str
    project: str
    version: str
    categories: List[Category]
    resolution: Optional[int]
    model_root_path: Optional[str]

    @property
    def context(self):
        return Context(organization=self.organization, project=self.project)

    @staticmethod
    def load_from_disk(model_root_path: str):
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
                model_information.model_root_path = model_root_path
            except Exception as e:
                raise Exception(
                    f"could not parse model information from file '{model_info_file_path}'. \n {str(e)}")

        return model_information

    def save(self):
        with open(self.model_root_path + '/model.json', 'w') as f:
            f.write(self.json(exclude={'model_root_path'}))
