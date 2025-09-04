import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union

from dacite import from_dict

from ..enums import CategoryType

KWONLY_SLOTS = {'kw_only': True, 'slots': True} if sys.version_info >= (3, 10) else {}


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
            logging.warning('could not find model information file %s', model_info_file_path)
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
class AboutResponse:
    operation_mode: str = field(metadata={"description": "The operation mode of the detector node"})
    state: Optional[str] = field(metadata={
        "description": "The state of the detector node",
        "example": "idle, online, detecting"})
    model_info: Optional[ModelInformation] = field(metadata={
        "description": "Information about the model of the detector node"})
    target_model: Optional[str] = field(metadata={"description": "The target model of the detector node"})
    version_control: str = field(metadata={
        "description": "The version control mode of the detector node",
        "example": "follow_loop, specific_version, pause"})


@dataclass(**KWONLY_SLOTS)
class ModelVersionResponse:
    current_version: str = field(metadata={"description": "The version of the model currently used by the detector."})
    target_version: str = field(metadata={"description": "The target model version set in the detector."})
    loop_version: str = field(metadata={"description": "The target model version specified by the loop."})
    local_versions: List[str] = field(metadata={"description": "The locally available versions of the model."})
    version_control: str = field(metadata={"description": "The version control mode."})


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
    state: NodeState = NodeState.Online
    uptime: int = 0
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
class DetectorStatus():
    uuid: str
    name: str
    state: NodeState
    uptime: int
    model_format: str
    current_model: Optional[str]
    target_model: Optional[str]
    errors: Dict
    operation_mode: str
