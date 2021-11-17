from typing import List, Optional
from pydantic import BaseModel
from enum import Enum
from learning_loop_node.trainer.model import Model
from typing import Union


class State(str, Enum):
    Idle = "idle"
    Offline = "offline"
    Online = "online"
    Running = "running"
    Preparing = "preparing"


class Status(BaseModel):
    id: str
    name: str
    state: Optional[State] = State.Offline
    uptime: Optional[int] = 0
    latest_error: Optional[str] = None

class AnnotationNodeStatus(Status):
    capabilities: List[str]

class TrainingStatus(Status):
    latest_produced_model_id: Optional[str]
    train_image_count: Optional[int]
    test_image_count: Optional[int]
    skipped_image_count: Optional[int]
