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
    id: str  # TODO: rename to identifier or uuid  (cannot be changed because of database / loop communication)
    name: str
    description: Optional[str] = None
    hotkey: Optional[str] = None
    color: Optional[str] = None
    point_size: Optional[int] = None
    # TODO: rename to ctype (cannot be changed because of database / loop communication)
    type: Optional[Union[CategoryType, str]] = None

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
    id: str
    host: Optional[str]
    organization: str
    project: str
    version: str
    categories: List[Category]
    resolution: Optional[int] = None
    model_root_path: Optional[str] = None
    model_size: Optional[str] = None

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
