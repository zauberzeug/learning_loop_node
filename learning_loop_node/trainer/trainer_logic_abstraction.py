from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

from ..data_classes import Context, Errors, PretrainedModel, TrainerState, TrainingData

if TYPE_CHECKING:
    from .trainer_node import TrainerNode


class TrainerLogicAbstraction(ABC):

    def __init__(self):
        self._node: Optional['TrainerNode'] = None  # type: ignore
        self.errors = Errors()

    @property
    def node(self) -> 'TrainerNode':
        assert self._node is not None, 'node should be set by TrainerNodes before initialization'
        return self._node

    @property
    @abstractmethod
    def state(self) -> TrainerState:
        """Returns the current state of the training logic"""

    @property
    @abstractmethod
    def training_uptime(self) -> float | None:
        """Returns the time in seconds since the training started or None if idle"""

    @property
    @abstractmethod
    def general_progress(self) -> float | None:
        """Returns the general progress of the training per state or None if idle"""

    @property
    @abstractmethod
    def provided_pretrained_models(self) -> List[PretrainedModel]:
        """Returns the list of provided pretrained models"""

    @property
    @abstractmethod
    def model_architecture(self) -> str:
        """Returns the architecture name of the model"""

    @property
    @abstractmethod
    def hyperparameters(self) -> dict | None:
        """Returns the hyperparameters if available"""

    @property
    @abstractmethod
    def training_data(self) -> TrainingData | None:
        """Returns the training data if available"""

    @property
    @abstractmethod
    def training_context(self) -> Context | None:
        """Returns the training context if available"""

    @abstractmethod
    async def begin_training(self, organization: str, project: str, details: dict):
        """Starts the training process"""

    @abstractmethod
    async def stop(self):
        """Stops the training process"""

    @abstractmethod
    async def shutdown(self):
        """Stops the training process and releases resources"""

    @abstractmethod
    async def continue_run_if_incomplete(self) -> bool:
        """Continues the training if it is incomplete"""
