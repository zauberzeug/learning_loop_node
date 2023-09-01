from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

# pylint: disable=no-name-in-module
from pydantic import BaseModel


class State(str, Enum):
    Idle = "idle"
    Offline = "offline"
    Online = "online"
    Preparing = "preparing"
    Running = "running"
    Stopping = "stopping"
    Detecting = 'detecting'
    Uploading = 'uploading'


class Status(BaseModel):
    id: str
    name: str
    state: Optional[State] = State.Offline
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


class AnnotationNodeStatus(Status):
    capabilities: List[str]


@dataclass
class DetectionStatus():
    id: str
    name: str
    state: Optional[State]
    errors: Optional[dict]
    uptime: Optional[int]

    model_format: str
    current_model: Optional[str]
    target_model: Optional[str]
    operation_mode: Optional[str]
