from typing import List, Optional
from dataclasses import dataclass, field
from ..status import State
from .model import PretrainedModel


@dataclass
class TrainingStatus():
    id: str
    name: str
    state: Optional[State]
    errors: Optional[dict]
    uptime: Optional[int]
    progress: Optional[float]

    train_image_count: Optional[int] = None
    test_image_count: Optional[int] = None
    skipped_image_count: Optional[int] = None
    pretrained_models: List[PretrainedModel] = field(default_factory=list)
    hyperparameters: Optional[str] = None
    architecture: Optional[str] = None
