from typing import List, Optional
from pydantic import BaseModel
from enum import Enum


class State(str, Enum):
    Idle = "idle"
    Running = "running"
    Offline = "offline"


class Status(BaseModel):
    id: str
    name: str
    state: Optional[State] = State.Offline
    uptime: Optional[int] = 0
    model: Optional[dict]
    hyperparameters: Optional[str]
    box_categories: Optional[dict]
    train_images: Optional[List[dict]]
    test_images: Optional[List[dict]]
