from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import os
from enum import Enum
from typing import List, Optional

# pylint: disable=no-name-in-module
from pydantic import BaseModel


class CategoryType(str, Enum):
    Box = 'box'
    Point = 'point'
    Segmentation = 'segmentation'
    Classification = 'classification'


class Category(BaseModel):
    identifier: str
    name: str
    description: Optional[str] = None
    hotkey: Optional[str] = None
    color: Optional[str] = None
    point_size: Optional[int] = None
    ctype: CategoryType = CategoryType.Box

    @staticmethod
    def from_list(values: List[dict]) -> List['Category']:
        categories: List[Category] = []
        for value in values:
            categories.append(Category.from_dict(value))
        return categories

    @staticmethod
    def from_dict(value: dict) -> 'Category':
        return Category.parse_obj(value)


def create_category(identifier: str, name: str, ctype: CategoryType):
    return Category(identifier=identifier, name=name, description='', hotkey='', color='', ctype=ctype, point_size=None)


class Context(BaseModel):
    organization: str
    project: str


# pylint: disable=no-name-in-module


class ModelInformation(BaseModel):
    id: str
    host: Optional[str]
    organization: str
    project: str
    version: str
    categories: List[Category]
    resolution: Optional[int] = None
    model_root_path: Optional[str] = None

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
            except Exception as exc:
                raise Exception(f"could not read model information from file '{model_info_file_path}'") from exc
            try:
                model_information = ModelInformation.parse_obj(content)
                model_information.model_root_path = model_root_path
            except Exception as exc:
                raise Exception(
                    f"could not parse model information from file '{model_info_file_path}'. \n {str(exc)}") from exc

        return model_information

    def save(self):
        if not self.model_root_path:
            raise Exception("model_root_path is not set")
        with open(self.model_root_path + '/model.json', 'w') as f:
            f.write(self.json(exclude={'model_root_path'}))


class ErrorConfiguration(BaseModel):
    begin_training: Optional[bool] = False
    save_model: Optional[bool] = False
    get_new_model: Optional[bool] = False
    crash_training: Optional[bool] = False


# pylint: disable=no-name-in-module


class NodeState(str, Enum):
    Idle = "idle"
    Offline = "offline"
    Online = "online"
    Preparing = "preparing"
    Running = "running"
    Stopping = "stopping"
    Detecting = 'detecting'
    Uploading = 'uploading'


class NodeStatus(BaseModel):
    id: str
    name: str
    state: Optional[NodeState] = NodeState.Offline
    uptime: Optional[int] = 0
    errors: Dict = {}

    def set_error(self, key: str, value: str):
        self.errors[key] = value

    def reset_error(self, key: str):
        try:
            del self.errors[key]
        except AttributeError:
            pass
        except KeyError:
            pass

    def reset_all_errors(self):
        for key in list(self.errors.keys()):
            self.reset_error(key)


class AnnotationNodeStatus(NodeStatus):
    capabilities: List[str]


@dataclass
class DetectionStatus():
    id: str
    name: str
    state: Optional[NodeState]
    errors: Optional[dict]
    uptime: Optional[int]

    model_format: str
    current_model: Optional[str]
    target_model: Optional[str]
    operation_mode: Optional[str]
