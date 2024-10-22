import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union

from dacite import from_dict

KWONLY_SLOTS = {'kw_only': True, 'slots': True} if sys.version_info >= (3, 10) else {}


class CategoryType(str, Enum):
    Box = 'box'
    Point = 'point'
    Segmentation = 'segmentation'
    Classification = 'classification'


@dataclass(**KWONLY_SLOTS)
class Category():
    id: str = field(metadata={"description": "The uuid of the category."})
    name: str = field(metadata={"description": "The name of the category."})
    description: Optional[str] = field(default=None, metadata={
        "description": "An optional description of the category."})
    hotkey: Optional[str] = field(default=None, metadata={
        "description": "The key shortcut of the category when annotating in the Learning Loop UI."})
    color: Optional[str] = field(default=None, metadata={
        "description": "The color of the category when displayed in the Learning Loop UI."})
    point_size: Optional[int] = field(default=None, metadata={
        "description": "The point size of the category in pixels. Represents the uncertainty of the category."})
    type: Optional[Union[CategoryType, str]] = field(default=None, metadata={
        "description": "The type of the category",
        "example": "box, point, segmentation, classification"})

    @staticmethod
    def from_list(values: List[dict]) -> List['Category']:
        return [from_dict(data_class=Category, data=value) for value in values]


@dataclass(**KWONLY_SLOTS)
class Context():
    organization: str
    project: str


# pylint: disable=no-name-in-module


@dataclass(**KWONLY_SLOTS)
class ModelInformation():
    id: str = field(metadata={"description": "The uuid of the model."})
    host: Optional[str] = field(metadata={"description": "The hostname that started the training.",
                                          "example": "learning-loop.ai"})
    organization: str = field(metadata={"description": "The owner organization of the model."})
    project: str = field(metadata={"description": "The project of the model."})
    version: str = field(metadata={"description": "The version of the model."})
    categories: List[Category] = field(default_factory=list, metadata={
                                       "description": "The categories used in the model."})
    resolution: Optional[int] = field(default=None, metadata={
        "description": "The resolution of the model (width and height of the image after preprocessing in pixels)."})
    model_root_path: Optional[str] = field(
        default=None, metadata={"description": "The path of the parent directory of the model in the file system."})
    model_size: Optional[str] = field(default=None, metadata={
                                      "description": "The size of the model (i.e. the specification or variant of the model architecture)."})

    @property
    def context(self):
        return Context(organization=self.organization, project=self.project)

    @staticmethod
    def load_from_disk(model_root_path: str) -> Optional['ModelInformation']:
        """Load model.json from model_root_path and return ModelInformation object.
        """
        model_info_file_path = f'{model_root_path}/model.json'
        if not os.path.exists(model_info_file_path):
            logging.warning(f"could not find model information file '{model_info_file_path}'")
            return None
        with open(model_info_file_path, 'r') as f:
            try:
                content = json.load(f)
            except Exception as exc:
                raise Exception(f"could not read model information from file '{model_info_file_path}'") from exc
            try:
                model_information = from_dict(data_class=ModelInformation, data=content)
                model_information.model_root_path = model_root_path
            except Exception as exc:
                raise Exception(
                    f"could not parse model information from file '{model_info_file_path}'. \n {str(exc)}") from exc

        return model_information

    def save(self):
        if not self.model_root_path:
            raise Exception("model_root_path is not set")
        with open(self.model_root_path + '/model.json', 'w') as f:
            self_as_dict = asdict(self)
            del self_as_dict['model_root_path']
            f.write(json.dumps(self_as_dict))

    @staticmethod
    def from_dict(data: Dict) -> 'ModelInformation':
        return from_dict(ModelInformation, data=data)


@dataclass(**KWONLY_SLOTS)
class ErrorConfiguration():
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


@dataclass(**KWONLY_SLOTS)
class NodeStatus():
    id: str
    name: str
    state: Optional[NodeState] = NodeState.Online
    uptime: Optional[int] = 0
    errors: Dict = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)

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


@dataclass(**KWONLY_SLOTS)
class AnnotationNodeStatus(NodeStatus):
    capabilities: List[str] = field(default_factory=list)


@dataclass(**KWONLY_SLOTS)
class DetectionStatus():
    id: str
    name: str
    model_format: str

    state: Optional[NodeState] = None
    errors: Optional[Dict] = None
    uptime: Optional[int] = None
    current_model: Optional[str] = None
    target_model: Optional[str] = None
    operation_mode: Optional[str] = None
